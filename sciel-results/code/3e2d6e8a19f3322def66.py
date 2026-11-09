import numpy as np
import random
import itertools
import time
import sys

def construct(q, n, radius):
    # Handle the specific task: q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: return all vertices (trivial cover)
        return [list(v) for v in itertools.product(range(q), repeat=n)]

    total_vertices = q ** n  # 729
    vertices = list(itertools.product(range(q), repeat=n))
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}

    # Precompute the closed balls (neighborhoods) for each vertex
    # For ternary length 6, each vertex has 1 + 6*(3-1) = 13 neighbors (including itself)
    balls = [set() for _ in range(total_vertices)]
    
    for i, v in enumerate(vertices):
        v_list = list(v)
        balls[i].add(i)
        for pos in range(n):
            old_val = v_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    v_list[pos] = new_val
                    idx = vertex_to_idx[tuple(v_list)]
                    balls[i].add(idx)
            v_list[pos] = old_val

    # Convert balls to bitsets for fast coverage checking
    # Use Python's arbitrary precision integers as bitsets
    ball_bitsets = [0] * total_vertices
    for i in range(total_vertices):
        mask = 0
        for j in balls[i]:
            mask |= (1 << j)
        ball_bitsets[i] = mask

    all_vertices_mask = (1 << total_vertices) - 1

    def is_dominating(S):
        """Check if set S is a dominating set using bitsets."""
        if not S:
            return total_vertices == 0
        covered = 0
        for idx in S:
            covered |= ball_bitsets[idx]
        return covered == all_vertices_mask

    def greedy_dominating_set(seed):
        """Greedy algorithm to find a dominating set."""
        random.seed(seed)
        undominated_mask = all_vertices_mask
        dominating_set = set()
        
        while undominated_mask:
            # Pick a random undominated vertex
            # Get list of undominated vertices
            undominated_list = []
            tmp = undominated_mask
            idx = 0
            while tmp:
                if tmp & 1:
                    undominated_list.append(idx)
                tmp >>= 1
                idx += 1
            
            target = random.choice(undominated_list)
            # Candidates are the ball of the target
            candidates = balls[target]
            
            # Pick the candidate that covers the most undominated vertices
            best_cand = None
            max_cover = -1
            for c in candidates:
                cover_count = bin(undominated_mask & ball_bitsets[c]).count('1')
                if cover_count > max_cover:
                    max_cover = cover_count
                    best_cand = c
            
            dominating_set.add(best_cand)
            undominated_mask &= ~ball_bitsets[best_cand]
            
        return dominating_set

    def local_search(S, time_limit=10):
        """
        Improves a dominating set S by removing redundant nodes and swapping.
        """
        S = set(S)
        start_time = time.time()
        
        # 1. Remove redundant nodes
        changed = True
        while changed:
            changed = False
            nodes = list(S)
            random.shuffle(nodes)
            for u in nodes:
                if u not in S:
                    continue
                new_S = S - {u}
                if is_dominating(new_S):
                    S = new_S
                    changed = True
        
        # 2. Local Search / Swaps
        # Try to swap nodes to potentially allow further removals
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            if not S:
                break
                
            u = random.choice(list(S))
            v = random.randint(0, total_vertices - 1)
            if v in S:
                continue
            
            new_S = (S - {u}) | {v}
            if is_dominating(new_S):
                S = new_S
                # Try to remove redundant nodes again
                changed = True
                while changed:
                    changed = False
                    nodes = list(S)
                    random.shuffle(nodes)
                    for w in nodes:
                        if w not in S:
                            continue
                        temp_S = S - {w}
                        if is_dominating(temp_S):
                            S = temp_S
                            changed = True

        return S

    def simulated_annealing(S, time_limit=30):
        """
        Simulated annealing to minimize dominating set size.
        Moves: remove a node, swap a node.
        Objective: minimize |S| subject to S being dominating.
        """
        S = set(S)
        current_size = len(S)
        best_S = set(S)
        best_size = current_size
        
        start_time = time.time()
        temp = 1.0
        cool_rate = 0.995
        
        while time.time() - start_time < time_limit:
            # Choose a move
            if not S:
                break
            
            move_type = random.random()
            if move_type < 0.3:
                # Try to remove a node
                u = random.choice(list(S))
                new_S = S - {u}
                if is_dominating(new_S):
                    delta = -1
                else:
                    delta = 1  # Penalty for breaking domination
            else:
                # Try to swap a node
                u = random.choice(list(S))
                v = random.randint(0, total_vertices - 1)
                if v in S:
                    continue
                new_S = (S - {u}) | {v}
                if is_dominating(new_S):
                    delta = 0
                else:
                    delta = 1  # Penalty for breaking domination
            
            # Accept/reject
            if delta < 0:
                S = new_S
                current_size = len(S)
                if current_size < best_size:
                    best_size = current_size
                    best_S = set(S)
            else:
                # Metropolis acceptance
                if random.random() < np.exp(-delta / temp):
                    S = new_S
                    current_size = len(S)
            
            temp *= cool_rate
            if temp < 0.01:
                break
        
        # Final cleanup
        S = best_S
        changed = True
        while changed:
            changed = False
            nodes = list(S)
            random.shuffle(nodes)
            for u in nodes:
                if u not in S:
                    continue
                new_S = S - {u}
                if is_dominating(new_S):
                    S = new_S
                    changed = True
        
        return S

    best_set = None
    best_size = float('inf')
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead
    
    seed = 0
    while time.time() - start_time < time_limit:
        # Generate initial solution
        S = greedy_dominating_set(seed)
        
        # Optimize with local search
        S = local_search(S, time_limit=5)
        
        # Further optimize with simulated annealing
        S = simulated_annealing(S, time_limit=20)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            # Early stop if we hit the theoretical lower bound or known best
            if best_size <= 73:
                break
        
        seed += 1

    if best_set is None:
        best_set = greedy_dominating_set(0)
        best_set = local_search(best_set, time_limit=5)
        best_set = simulated_annealing(best_set, time_limit=20)

    return [vertices[idx] for idx in best_set]
