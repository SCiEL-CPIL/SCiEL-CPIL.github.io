"""Rewritten HP lattice protein folding using Tabu Search."""
import random
import math
from collections import deque


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the negative H-H contact energy."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _initial_spiral(sequence, dimension):
    """Create a compact initial conformation using a spiral-like pattern."""
    n = len(sequence)
    if n <= 1:
        return [(0,) * dimension]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    direction = (1, 0)
    
    for i in range(1, n):
        placed = False
        for turn in (1, 0, -1):
            if turn == 1:
                new_dir = (direction[1], -direction[0])
            elif turn == -1:
                new_dir = (-direction[1], direction[0])
            else:
                new_dir = direction
            
            new_pos = (chain[-1][0] + new_dir[0], chain[-1][1] + new_dir[1])
            if new_pos not in occupied:
                chain.append(new_pos)
                occupied[new_pos] = i
                direction = new_dir
                placed = True
                break
        
        if not placed:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_pos = (chain[-1][0] + dx, chain[-1][1] + dy)
                if new_pos not in occupied:
                    chain.append(new_pos)
                    occupied[new_pos] = i
                    direction = (dx, dy)
                    placed = True
                    break
        
        if not placed:
            for j in range(i, n):
                chain.append((chain[-1][0], chain[-1][1] + 1))
            break
    
    return chain


def _tabu_search(sequence, dimension, rng, num_iterations=2000, tabu_length=50):
    """Tabu Search with single-step and reversal moves."""
    n = len(sequence)
    if n <= 1:
        return _initial_spiral(sequence, dimension)
    
    chain = _initial_spiral(sequence, dimension)
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _compute_energy(sequence, chain)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    tabu_list = deque(maxlen=tabu_length)
    
    no_improve_counter = 0
    max_no_improve = 1000
    
    for it in range(num_iterations):
        if no_improve_counter > max_no_improve:
            break
            
        candidates = []
        
        # 1. Single step move candidate
        if n > 2:
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            for q in _neighbours(old_pos):
                if q == old_pos:
                    continue
                if q in occupied and occupied[q] != i:
                    continue
                if abs(q[0] - chain[i-1][0]) + abs(q[1] - chain[i-1][1]) != 1:
                    continue
                if abs(q[0] - chain[i+1][0]) + abs(q[1] - chain[i+1][1]) != 1:
                    continue
                
                candidates.append(('step', i, old_pos, q))
        
        # 2. Reversal move candidate
        if n > 4:
            seg_len = rng.randint(2, min(10, n - 2))
            i = rng.randint(0, n - seg_len - 1)
            j = i + seg_len
            
            valid = True
            if i > 0:
                prev = chain[i - 1]
                new_first = chain[j]
                if abs(prev[0] - new_first[0]) + abs(prev[1] - new_first[1]) != 1:
                    valid = False
            if j < n - 1:
                next_pos = chain[j + 1]
                new_last = chain[i]
                if abs(next_pos[0] - new_last[0]) + abs(next_pos[1] - new_last[1]) != 1:
                    valid = False
            
            if valid:
                candidates.append(('rev', i, j, None))
        
        if not candidates:
            no_improve_counter += 1
            continue
        
        best_candidate = None
        best_candidate_energy = float('inf')
        
        for move in candidates:
            move_type = move[0]
            
            # Check tabu status
            is_tabu = False
            for entry in tabu_list:
                if move_type == 'step':
                    i = move[1]
                    old_pos = move[2]
                    new_pos = move[3]
                    # Tabu if we recently moved i from old_pos to new_pos
                    if entry[0] == 'step' and entry[1] == i and entry[2] == old_pos and entry[3] == new_pos:
                        is_tabu = True
                        break
                elif move_type == 'rev':
                    i = move[1]
                    j = move[2]
                    if entry[0] == 'rev' and entry[1] == i and entry[2] == j:
                        is_tabu = True
                        break
            
            # Apply move temporarily to get energy
            if move_type == 'step':
                i = move[1]
                old_pos = move[2]
                new_pos = move[3]
                
                chain[i] = new_pos
                del occupied[old_pos]
                occupied[new_pos] = i
                
                new_energy = _compute_energy(sequence, chain)
                
                chain[i] = old_pos
                del occupied[new_pos]
                occupied[old_pos] = i
                
            elif move_type == 'rev':
                i = move[1]
                j = move[2]
                
                old_segment = chain[i:j + 1]
                chain[i:j + 1] = old_segment[::-1]
                for k in range(i, j + 1):
                    occupied[chain[k]] = k
                
                new_energy = _compute_energy(sequence, chain)
                
                chain[i:j + 1] = old_segment
                for k in range(i, j + 1):
                    occupied[chain[k]] = k
            
            if new_energy < best_candidate_energy:
                best_candidate_energy = new_energy
                best_candidate = move
                best_candidate_is_tabu = is_tabu
        
        if best_candidate is None:
            continue
        
        # Aspiration criterion
        if best_candidate_is_tabu and best_candidate_energy >= best_energy:
            continue
        
        # Apply the best candidate
        move_type = best_candidate[0]
        if move_type == 'step':
            i = best_candidate[1]
            old_pos = best_candidate[2]
            new_pos = best_candidate[3]
            
            chain[i] = new_pos
            del occupied[old_pos]
            occupied[new_pos] = i
            
            tabu_list.append(('step', i, old_pos, new_pos))
            
        elif move_type == 'rev':
            i = best_candidate[1]
            j = best_candidate[2]
            
            old_segment = chain[i:j + 1]
            chain[i:j + 1] = old_segment[::-1]
            for k in range(i, j + 1):
                occupied[chain[k]] = k
            
            tabu_list.append(('rev', i, j, None))
        
        current_energy = best_candidate_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(chain)
            no_improve_counter = 0
        else:
            no_improve_counter += 1
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using Tabu Search with multiple restarts."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    best_chain = None
    best_energy = 0
    
    num_restarts = 5
    
    for restart in range(num_restarts):
        rng = random.Random(12345 + len(sequence) * 100 + restart)
        
        if n <= 30:
            num_iter = 3000
        elif n <= 60:
            num_iter = 2000
        else:
            num_iter = 1500
            
        chain = _tabu_search(sequence, dimension, rng, num_iter)
        energy = _compute_energy(sequence, chain)
        
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
