import time
import random
import sys
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 85.0

def _build_subset_trans(trans):
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset

def compute_threshold(trans_a, trans_b):
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)
    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
    return 0

def _generate_structural_candidates():
    """
    Generate diverse structural candidates based on known high-threshold patterns:
    1. Pinwheel: a=cycle, b=comb with one defect
    2. Ping-pong: two interleaved cycles with defects
    3. Random permutations with single defects
    """
    candidates = []
    
    # Pattern 1: Pinwheel variants
    a_cycle = [(i + 1) % N for i in range(N)]
    for defect_pos in range(N):
        for defect_dest in range(N):
            if defect_dest == (defect_pos + 1) % N:
                continue  # skip if no actual defect
            b = [(i + 1) % N for i in range(N)]
            b[defect_pos] = defect_dest
            candidates.append((a_cycle[:], b[:]))
    
    # Pattern 2: Two cycles with defects
    for split in range(2, N):
        for defect_pos in range(N):
            a = [0] * N
            b = [0] * N
            # a: two cycles
            for i in range(split):
                a[i] = (i + 1) % split
            for i in range(split, N):
                a[i] = split + (i - split + 1) % (N - split)
            # b: different split with defect
            b_split = N - split
            for i in range(b_split):
                b[i] = (i + 1) % b_split
            for i in range(b_split, N):
                b[i] = b_split + (i - b_split + 1) % (N - b_split)
            b[defect_pos] = random.randint(0, N - 1)
            candidates.append((a[:], b[:]))
    
    # Pattern 3: Random permutations with single defects
    for _ in range(50):
        a = list(range(N))
        random.shuffle(a)
        b = list(range(N))
        random.shuffle(b)
        # Add a defect to ensure synchronizability
        defect_pos = random.randint(0, N - 1)
        a[defect_pos] = random.randint(0, N - 1)
        candidates.append((a[:], b[:]))
    
    return candidates

def _local_search(a, b, start_time, deadline, max_no_improve=30):
    """
    Hill climbing with targeted mutations:
    - Defect adjustment: modify the "broken" state in a cycle
    - Asymmetry: swap destinations to create asymmetry
    - Random: single-state random mutation
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    
    if cur_score == 0:
        return None, None, 0
        
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    no_improve = 0
    
    while time.time() < deadline and no_improve < max_no_improve:
        mutation_type = random.choice(['defect', 'asymmetry', 'random'])
        
        if mutation_type == 'defect':
            # Find a state that breaks a cycle and modify it
            # Heuristic: states where a[i] != (i+1)%N are "defect" candidates
            defect_states = [i for i in range(N) if cur_a[i] != (i + 1) % N]
            if not defect_states:
                defect_states = list(range(N))
            
            state = random.choice(defect_states)
            old = cur_a[state]
            new_dest = random.randint(0, N - 1)
            if new_dest == old:
                no_improve += 1
                continue
                
            cur_a[state] = new_dest
            new_score = compute_threshold(cur_a, cur_b)
            
            if new_score > cur_score:
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_a = cur_a[:]
                    best_b = cur_b[:]
            else:
                cur_a[state] = old
                no_improve += 1
                
        elif mutation_type == 'asymmetry':
            # Swap destinations of two states in a or b
            i, j = random.sample(range(N), 2)
            if random.random() < 0.5:
                old_ai, old_aj = cur_a[i], cur_a[j]
                cur_a[i], cur_a[j] = old_aj, old_ai
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_a[i], cur_a[j] = old_ai, old_aj
                    no_improve += 1
            else:
                old_bi, old_bj = cur_b[i], cur_b[j]
                cur_b[i], cur_b[j] = old_bj, old_bi
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_b[i], cur_b[j] = old_bi, old_bj
                    no_improve += 1
                    
        else:  # random
            state = random.randint(0, N - 1)
            if random.random() < 0.5:
                old = cur_a[state]
                new_dest = random.randint(0, N - 1)
                if new_dest == old:
                    no_improve += 1
                    continue
                cur_a[state] = new_dest
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_a[state] = old
                    no_improve += 1
            else:
                old = cur_b[state]
                new_dest = random.randint(0, N - 1)
                if new_dest == old:
                    no_improve += 1
                    continue
                cur_b[state] = new_dest
                new_score = compute_threshold(cur_a, cur_b)
                if new_score > cur_score:
                    cur_score = new_score
                    no_improve = 0
                    if cur_score > best_score:
                        best_score = cur_score
                        best_a = cur_a[:]
                        best_b = cur_b[:]
                else:
                    cur_b[state] = old
                    no_improve += 1
                    
    return best_a, best_b, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    random.seed(123)
    
    best_a = None
    best_b = None
    best_score = 0
    
    # Phase 1: Evaluate structural candidates
    candidates = _generate_structural_candidates()
    
    for a_cand, b_cand in candidates:
        if time.time() > deadline - 15.0:
            break
        score = compute_threshold(a_cand, b_cand)
        if score > best_score:
            best_score = score
            best_a = a_cand[:]
            best_b = b_cand[:]
            
    if best_score == (N - 1) ** 2:
        return [best_a, best_b]
    
    # Phase 2: Local search from best candidate
    if best_a is not None:
        refined_a, refined_b, refined_score = _local_search(
            best_a, best_b, start_time, deadline - 5.0
        )
        if refined_score > best_score:
            best_score = refined_score
            best_a = refined_a
            best_b = refined_b
            
    if best_score == (N - 1) ** 2:
        return [best_a, best_b]
    
    # Phase 3: Random restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            
            # Perturb best known
            a_pert = best_a[:]
            b_pert = best_b[:]
            for _ in range(2):
                if random.random() < 0.5:
                    i = random.randint(0, N - 1)
                    a_pert[i] = random.randint(0, N - 1)
                else:
                    i = random.randint(0, N - 1)
                    b_pert[i] = random.randint(0, N - 1)
            
            score = compute_threshold(a_pert, b_pert)
            if score > best_score:
                best_score = score
                best_a = a_pert[:]
                best_b = b_pert[:]
                
            if score >= best_score - 3:
                refined_a, refined_b, refined_score = _local_search(
                    a_pert, b_pert, start_time, deadline - 2.0, max_no_improve=20
                )
                if refined_score > best_score:
                    best_score = refined_score
                    best_a = refined_a
                    best_b = refined_b
                    
            if best_score == (N - 1) ** 2:
                return [best_a, best_b]
    
    # Fallback
    if best_score == 0 or best_a is None:
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[0] = 0
        return [a, b]
    
    return [best_a, best_b]

if __name__ == "__main__":
    a, b = construct()
    print(f"Best score found: {compute_threshold(a, b)}")
    print(f"A: {a}")
    print(f"B: {b}")
