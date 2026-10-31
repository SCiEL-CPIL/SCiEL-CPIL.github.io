import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, compute its t-subsets
    block_tsubs = []
    for b in all_blocks:
        block_tsubs.append(frozenset(itertools.combinations(b, t)))
    
    # Map each t-subset to blocks containing it
    tsub_to_blocks = defaultdict(list)
    for i, tsubs in enumerate(block_tsubs):
        for ts in tsubs:
            tsub_to_blocks[ts].append(i)
    
    # Count how many blocks contain each point (for diversity)
    point_frequency = np.zeros(v, dtype=np.int32)
    for b in all_blocks:
        for p in b:
            point_frequency[p] += 1
    
    # Simulated annealing approach
    # State: set of selected block indices
    # Energy: number of blocks (we want to minimize)
    # Moves: add/remove/swap blocks
    
    best_solution = None
    best_energy = float('inf')
    
    # Multiple restarts with different initial solutions
    num_restarts = 3
    restarts_done = 0
    
    while restarts_done < num_restarts and time.time() < deadline - 30:
        restarts_done += 1
        current_time = time.time()
        restart_deadline = deadline - 30 - (restarts_done - 1) * 30
        
        # Initial solution: greedy
        selected = set()
        coverage = defaultdict(int)
        
        # Greedy: repeatedly add block that covers most uncovered t-subsets
        while time.time() < restart_deadline - 20:
            best_gain = -1
            best_block = None
            for i in range(num_blocks):
                if i in selected:
                    continue
                gain = 0
                for ts in block_tsubs[i]:
                    if coverage[ts] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = i
            if best_block is None or best_gain <= 0:
                break
            selected.add(best_block)
            for ts in block_tsubs[best_block]:
                coverage[ts] += 1
        
        current_energy = len(selected)
        
        # Simulated annealing refinement
        T = 10.0  # Temperature
        T_min = 0.01
        cooling_rate = 0.999
        max_iterations = 5000
        iterations = 0
        
        while T > T_min and iterations < max_iterations and time.time() < restart_deadline - 5:
            iterations += 1
            move_type = random.random()
            
            if move_type < 0.4:
                # Remove a random block
                if not selected:
                    continue
                to_remove = random.choice(list(selected))
                new_energy = current_energy - 1
                # Check if removal breaks coverage
                valid = True
                for ts in block_tsubs[to_remove]:
                    if coverage[ts] <= 1:
                        valid = False
                        break
                if not valid:
                    continue
                # Accept with probability based on energy change
                delta_E = new_energy - current_energy
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    selected.remove(to_remove)
                    for ts in block_tsubs[to_remove]:
                        coverage[ts] -= 1
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = set(selected)
                        
            elif move_type < 0.7:
                # Add a random block
                to_add = random.randint(0, num_blocks - 1)
                if to_add in selected:
                    continue
                new_energy = current_energy + 1
                delta_E = new_energy - current_energy
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    selected.add(to_add)
                    for ts in block_tsubs[to_add]:
                        coverage[ts] += 1
                    current_energy = new_energy
                    
            else:
                # Swap: remove one, add one
                if not selected:
                    continue
                to_remove = random.choice(list(selected))
                to_add = random.randint(0, num_blocks - 1)
                if to_add in selected:
                    continue
                # Check if swap maintains coverage
                valid = True
                for ts in block_tsubs[to_remove]:
                    if coverage[ts] <= 1 and to_add not in tsub_to_blocks[ts]:
                        valid = False
                        break
                if not valid:
                    continue
                new_energy = current_energy  # Same size
                delta_E = 0
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    selected.remove(to_remove)
                    selected.add(to_add)
                    for ts in block_tsubs[to_remove]:
                        coverage[ts] -= 1
                    for ts in block_tsubs[to_add]:
                        coverage[ts] += 1
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = set(selected)
            
            T *= cooling_rate
        
        # Local optimization: try to remove redundant blocks
        changed = True
        while changed and time.time() < restart_deadline - 3:
            changed = False
            for block_idx in list(selected):
                is_redundant = True
                for ts in block_tsubs[block_idx]:
                    # Count how many selected blocks cover this t-subset
                    count = 0
                    for other_idx in tsub_to_blocks[ts]:
                        if other_idx in selected:
                            count += 1
                            if count > 1:
                                break
                    if count <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    selected.remove(block_idx)
                    for ts in block_tsubs[block_idx]:
                        coverage[ts] -= 1
                    current_energy = len(selected)
                    changed = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_solution = set(selected)
    
    # Return best solution found
    if best_solution is None:
        # Fallback: return a valid (possibly large) solution
        selected = set(range(num_blocks))
        return [list(all_blocks[i]) for i in sorted(selected)]
    else:
        return [list(all_blocks[i]) for i in sorted(best_solution)]
