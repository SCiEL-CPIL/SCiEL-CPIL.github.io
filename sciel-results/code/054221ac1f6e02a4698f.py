"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _bfs_reset_length_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Uses integer bitmasks and precomputed lookup tables for speed.
    """
    N = len(a_trans)
    start_mask = (1 << N) - 1
    
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0
    
    # Precompute: for each bit position i, what's the destination bit for a and b
    # We need to map from a mask to the next mask efficiently
    # Strategy: precompute for each possible subset (up to 2^12 = 4096) the next state
    # This is feasible for N=12
    
    # Precompute transition tables for all 2^N subsets
    num_subsets = 1 << N
    trans_a_table = [0] * num_subsets
    trans_b_table = [0] * num_subsets
    
    # For each subset, compute where a and b take it
    for mask in range(num_subsets):
        next_a = 0
        next_b = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            next_a |= (1 << a_trans[idx])
            next_b |= (1 << b_trans[idx])
            m ^= lsb
        trans_a_table[mask] = next_a
        trans_b_table[mask] = next_b
    
    visited = [False] * num_subsets
    visited[start_mask] = True
    queue = deque([(start_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        
        if mask & (mask - 1) == 0:
            return dist
        
        next_a = trans_a_table[mask]
        if not visited[next_a]:
            visited[next_a] = True
            queue.append((next_a, dist + 1))
        
        next_b = trans_b_table[mask]
        if not visited[next_b]:
            visited[next_b] = True
            queue.append((next_b, dist + 1))
    
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a_trans, b_trans):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return
        score = _bfs_reset_length_fast(a_trans, b_trans)
        if score > best_score:
            best_score = score
            best_a = a_trans[:]
            best_b = b_trans[:]
            return True
        return False

    # Strategy 1: Pinwheel constructions with different parameters
    for step in range(1, N):
        a_trans = [(i + step) % N for i in range(N)]
        # b as reverse cycle with one break
        for break_pos in range(N):
            for target in range(N):
                b_trans = [(i - 1) % N for i in range(N)]
                b_trans[break_pos] = target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline:
                    return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
        if time.time() > deadline:
            return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Strategy 2: Rank-2 constructions - a is a cycle, b maps most to one state
    for cycle_step in range(1, N):
        a_trans = [(i + cycle_step) % N for i in range(N)]
        # b maps all but one state to a common target
        for excluded in range(N):
            for common_target in range(N):
                b_trans = [common_target] * N
                b_trans[excluded] = (excluded + cycle_step) % N  # Keep one state in cycle
                try_automaton(a_trans, b_trans)
                if time.time() > deadline:
                    return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
        if time.time() > deadline:
            return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Strategy 3: a is cycle, b is "almost identity" with one or two breaks
    for cycle_step in range(1, N):
        a_trans = [(i + cycle_step) % N for i in range(N)]
        # b with one break
        for break_pos in range(N):
            for target in range(N):
                b_trans = list(range(N))
                b_trans[break_pos] = target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline:
                    return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
        # b with two breaks
        for break1 in range(N):
            for break2 in range(N):
                if break1 == break2:
                    continue
                for t1 in range(N):
                    for t2 in range(N):
                        b_trans = list(range(N))
                        b_trans[break1] = t1
                        b_trans[break2] = t2
                        try_automaton(a_trans, b_trans)
                        if time.time() > deadline:
                            return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
        if time.time() > deadline:
            return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Strategy 4: a is cycle, b is "almost reverse cycle" with breaks
    for cycle_step in range(1, N):
        a_trans = [(i + cycle_step) % N for i in range(N)]
        # b as (i-1) mod N with one break
        for break_pos in range(N):
            for target in range(N):
                b_trans = [(i - 1) % N for i in range(N)]
                b_trans[break_pos] = target
                try_automaton(a_trans, b_trans)
                if time.time() > deadline:
                    return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
        if time.time() > deadline:
            return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Strategy 5: Both a and b are structured permutations with breaks
    # a: cycle with one break, b: cycle with one break
    for a_cycle_step in range(1, N):
        a_trans = [(i + a_cycle_step) % N for i in range(N)]
        for a_break in range(N):
            for a_target in range(N):
                a_trans[a_break] = a_target
                for b_cycle_step in range(1, N):
                    b_trans = [(i + b_cycle_step) % N for i in range(N)]
                    for b_break in range(N):
                        for b_target in range(N):
                            b_trans[b_break] = b_target
                            try_automaton(a_trans, b_trans)
                            if time.time() > deadline:
                                return [best_a, best_b] if best_a else [[(i + 1) % N for i in range(N)], [0] * N]
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Strategy 6: Enhanced Local Search with structured mutations
    if time.time() < deadline - 3.0:
        random.seed(42)
        
        if best_a is None:
            current_a = [(i + 1) % N for i in range(N)]
            current_b = [(i - 1) % N for i in range(N)]
        else:
            current_a = best_a[:]
            current_b = best_b[:]
        
        current_score = best_score
        no_improve_count = 0
        max_no_improve = 300
        
        while time.time() < deadline - 2.0:
            if no_improve_count > max_no_improve:
                # Restart: perturb with 2-4 random changes
                for _ in range(random.randint(2, 4)):
                    if random.random() < 0.5:
                        i = random.randint(0, N - 1)
                        current_a[i] = random.randint(0, N - 1)
                    else:
                        i = random.randint(0, N - 1)
                        current_b[i] = random.randint(0, N - 1)
                
                new_score = _bfs_reset_length_fast(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    current_a = best_a[:]
                    current_b = best_b[:]
                    current_score = best_score
                
                no_improve_count = 0
                max_no_improve = random.randint(300, 600)
                continue
            
            # Mutation: try different types
            mutation_type = random.random()
            
            if mutation_type < 0.3:
                # Single flip in a
                i = random.randint(0, N - 1)
                old = current_a[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old:
                    continue
                current_a[i] = new_dest
                new_score = _bfs_reset_length_fast(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                    no_improve_count = 0
                else:
                    current_a[i] = old
                    no_improve_count += 1
                    
            elif mutation_type < 0.5:
                # Single flip in b
                i = random.randint(0, N - 1)
                old = current_b[i]
                new_dest = random.randint(0, N - 1)
                if new_dest == old:
                    continue
                current_b[i] = new_dest
                new_score = _bfs_reset_length_fast(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                    no_improve_count = 0
                else:
                    current_b[i] = old
                    no_improve_count += 1
                    
            elif mutation_type < 0.7:
                # Swap in a
                i, j = random.sample(range(N), 2)
                current_a[i], current_a[j] = current_a[j], current_a[i]
                new_score = _bfs_reset_length_fast(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                    no_improve_count = 0
                else:
                    current_a[i], current_a[j] = current_a[j], current_a[i]
                    no_improve_count += 1
                    
            elif mutation_type < 0.85:
                # Swap in b
                i, j = random.sample(range(N), 2)
                current_b[i], current_b[j] = current_b[j], current_b[i]
                new_score = _bfs_reset_length_fast(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                    no_improve_count = 0
                else:
                    current_b[i], current_b[j] = current_b[j], current_b[i]
                    no_improve_count += 1
            else:
                # Double flip: change two positions (one in a, one in b)
                i1 = random.randint(0, N - 1)
                i2 = random.randint(0, N - 1)
                old1 = current_a[i1]
                old2 = current_b[i2]
                new1 = random.randint(0, N - 1)
                new2 = random.randint(0, N - 1)
                current_a[i1] = new1
                current_b[i2] = new2
                new_score = _bfs_reset_length_fast(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                    no_improve_count = 0
                else:
                    current_a[i1] = old1
                    current_b[i2] = old2
                    no_improve_count += 1

    # Fallback: ensure we return a valid synchronizing automaton
    if best_a is None:
        # Simple pinwheel that definitely synchronizes
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
