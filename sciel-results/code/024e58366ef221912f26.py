import time
import random
from collections import deque
import numpy as np

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact reset threshold using BFS on the subset automaton.
    Returns the length of the shortest reset word.
    """
    full_set = (1 << N) - 1
    visited = {full_set: 0}
    queue = deque([(full_set, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask.bit_count() == 1:
            return dist
        
        new_mask_a = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= (1 << trans_a[i])
        
        if new_mask_a not in visited:
            visited[new_mask_a] = dist + 1
            queue.append((new_mask_a, dist + 1))
            
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= (1 << trans_b[i])
        
        if new_mask_b not in visited:
            visited[new_mask_b] = dist + 1
            queue.append((new_mask_b, dist + 1))
            
    return -1

def construct():
    """
    Multi-start hill-climb with vectorized NumPy BFS.
    Explores diverse structural families and accepts non-worsening moves.
    """
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s kill

    def fast_threshold_np(a_list, b_list):
        """
        Vectorized BFS using NumPy. Precomputes transition tables for all 4096 masks,
        then performs BFS level-by-level with vectorized operations.
        """
        a_arr = np.array(a_list, dtype=np.int32)
        b_arr = np.array(b_list, dtype=np.int32)
        
        # Precompute: for each mask (0..4095), compute image under a and b
        # mask has bits 0..11 set. Image = OR of (1 << trans[i]) for each set bit i.
        # We can build this incrementally: image[mask] = image[mask without lowest set bit] | (1 << trans[lowest_bit])
        
        all_masks = np.arange(1 << N, dtype=np.int64)
        img_a = np.zeros(1 << N, dtype=np.int64)
        img_b = np.zeros(1 << N, dtype=np.int64)
        
        for mask in range(1, 1 << N):
            # Get lowest set bit
            lowbit = mask & (-mask)
            bit_idx = lowbit.bit_length() - 1
            prev = mask ^ lowbit
            img_a[mask] = img_a[prev] | (1 << a_arr[bit_idx])
            img_b[mask] = img_b[prev] | (1 << b_arr[bit_idx])
        
        full_set = (1 << N) - 1
        visited = np.zeros(1 << N, dtype=bool)
        visited[full_set] = True
        
        current_frontier = np.array([full_set], dtype=np.int64)
        dist = 0
        
        while len(current_frontier) > 0:
            # Check if any mask in frontier is a singleton
            popcounts = np.array([int(m).bit_count() for m in current_frontier])
            if np.any(popcounts == 1):
                return dist
            
            # Generate next frontier: images under a and b
            next_a = img_a[current_frontier]
            next_b = img_b[current_frontier]
            candidates = np.concatenate([next_a, next_b])
            
            # Filter out already visited
            new_candidates = candidates[~visited[candidates]]
            if len(new_candidates) == 0:
                break
            
            # Deduplicate
            new_candidates = np.unique(new_candidates)
            visited[new_candidates] = True
            current_frontier = new_candidates
            dist += 1
        
        return -1

    def make_cerny_variant():
        """Černý-type: a is a cycle, b resets one state."""
        a = [(i + 1) % N for i in range(N)]
        b = [i - 1 if i > 0 else 0 for i in range(N)]
        return a, b

    def make_ping_pong():
        """Ping-pong: a and b are partial reverses with a trap."""
        a = list(range(N))
        b = list(range(N))
        # Make a cycle: 0->1->2->...->N-1->0
        a = [(i + 1) % N for i in range(N)]
        # b: mostly identity, but one state goes to 0
        b = [i for i in range(N)]
        b[N - 1] = 0
        return a, b

    def make_random():
        """Random automaton."""
        a = [random.randint(0, N - 1) for _ in range(N)]
        b = [random.randint(0, N - 1) for _ in range(N)]
        return a, b

    # Initialize with multiple diverse starting points
    initializations = [
        make_cerny_variant(),
        make_ping_pong(),
        make_random(),
        make_random(),
        make_random(),
    ]

    best_a = None
    best_b = None
    best_score = -1

    random.seed(42)

    for init_a, init_b in initializations:
        if time.time() - start_time > deadline:
            break

        current_a = init_a[:]
        current_b = init_b[:]
        current_score = fast_threshold_np(current_a, current_b)

        if current_score < 0:
            continue  # Not synchronizing, skip

        if current_score > best_score:
            best_score = current_score
            best_a = current_a[:]
            best_b = current_b[:]

        # Hill-climb from this starting point
        stagnation_count = 0
        max_stagnation = 50

        while time.time() - start_time < deadline and stagnation_count < max_stagnation:
            # Random mutation
            letter = random.randint(0, 1)
            state = random.randint(0, N - 1)
            old_dest = current_a[state] if letter == 0 else current_b[state]
            new_dest = random.randint(0, N - 1)

            if new_dest == old_dest:
                continue

            if letter == 0:
                current_a[state] = new_dest
            else:
                current_b[state] = new_dest

            new_score = fast_threshold_np(current_a, current_b)

            if new_score > current_score:
                current_score = new_score
                stagnation_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # Revert (accept non-worsening only for diversity, but here we revert)
                if letter == 0:
                    current_a[state] = old_dest
                else:
                    current_b[state] = old_dest
                stagnation_count += 1

    if best_a is None:
        # Fallback: return the Černý automaton
        best_a, best_b = make_cerny_variant()

    return [best_a, best_b]

# EVOLVE-BLOCK-END
