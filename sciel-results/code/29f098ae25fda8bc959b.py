import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n <= 4:
        # For small n, the standard construction is optimal or near-optimal and fast
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 5, use heuristic search
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # overlap[i][j] is the max k such that perms[i][n-k:] == perms[j][:k]
    # Max k is n-1
    overlap = [[0] * N for _ in range(N)]
    for i in range(N):
        p1 = perms[i]
        for j in range(N):
            p2 = perms[j]
            # Check overlaps from n-1 down to 1
            # If p1[1:] == p2[:-1], overlap is n-1
            # We can optimize by checking suffix/prefix
            # Since n is small (7), linear check is fine
            max_k = 0
            for k in range(n - 1, 0, -1):
                if p1[n-k:] == p2[:k]:
                    max_k = k
                    break
            overlap[i][j] = max_k

    # Get initial sequence from standard superpermutation
    def get_standard_sequence(n):
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                ov = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[ov:]
            word = new
        # Extract permutations from the final word
        seq = []
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if len(set(p)) == n:
                seq.append(perm_to_idx[p])
        # Ensure all perms are present
        if len(seq) != N:
            # Fallback to random if extraction fails (should not happen)
            return list(range(N))
        return seq

    seq = get_standard_sequence(n)
    
    # Simulated Annealing
    # Objective: Maximize sum of overlaps
    # Cost = -Sum(overlap)
    def calc_total_overlap(s):
        total = 0
        for i in range(len(s) - 1):
            total += overlap[s[i]][s[i+1]]
        return total

    current_overlap = calc_total_overlap(seq)
    best_seq = seq[:]
    best_overlap = current_overlap
    
    T = 1.0
    cooling = 0.9999
    min_T = 1e-4
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick two random indices to swap
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
            
        # Calculate delta
        # Neighbors of i: i-1, i+1
        # Neighbors of j: j-1, j+1
        # If i and j are adjacent, handle specially
        
        old_cost = 0
        new_cost = 0
        
        # Indices to check
        idxs = [i-1, i, i+1, j-1, j, j+1]
        # Filter valid indices
        valid_idxs = [x for x in idxs if 0 <= x < N]
        
        # It's easier to just calculate the change in the specific edges
        # Edges involved:
        # (i-1, i), (i, i+1)
        # (j-1, j), (j, j+1)
        # If i and j are adjacent, edges overlap.
        
        def get_ov(a, b):
            if 0 <= a < N and 0 <= b < N:
                return overlap[a][b]
            return 0
            
        # Old edges
        old_edges = set()
        if i > 0: old_edges.add((i-1, i))
        if i < N-1: old_edges.add((i, i+1))
        if j > 0: old_edges.add((j-1, j))
        if j < N-1: old_edges.add((j, j+1))
        
        # New edges after swap
        # i has perm[j], j has perm[i]
        new_edges = set()
        if i > 0: new_edges.add((i-1, j)) # seq[i-1] -> seq[j] (which is now at i)
        if i < N-1: new_edges.add((j, i+1)) # seq[j] (at i) -> seq[i+1]
        if j > 0: new_edges.add((j-1, i)) # seq[j-1] -> seq[i] (which is now at j)
        if j < N-1: new_edges.add((i, j+1)) # seq[i] (at j) -> seq[j+1]
        
        # If i and j are adjacent, e.g., j = i+1
        # Old: (i-1, i), (i, i+1), (i+1, i+2)
        # New: (i-1, i+1), (i+1, i), (i, i+2)
        # The set logic above might double count or miss if not careful.
        # Let's just compute sum of overlaps for the affected edges directly.
        
        old_sum = 0
        if i > 0: old_sum += overlap[seq[i-1]][seq[i]]
        if i < N-1: old_sum += overlap[seq[i]][seq[i+1]]
        if j > 0: old_sum += overlap[seq[j-1]][seq[j]]
        if j < N-1: old_sum += overlap[seq[j]][seq[j+1]]
        
        # If i and j are adjacent, we counted the middle edge twice?
        # If j = i+1:
        # Old edges: (i-1, i), (i, i+1), (i+1, i+2)
        # My loop:
        # i>0: (i-1, i)
        # i<N-1: (i, i+1)
        # j>0: (i, i+1) -> Duplicate!
        # j<N-1: (i+1, i+2)
        # So if adjacent, subtract the middle edge once.
        if j == i + 1 or i == j + 1:
            old_sum -= overlap[seq[min(i,j)]][seq[max(i,j)]]
            
        new_sum = 0
        # Swap values in a temporary view
        pi = seq[i]
        pj = seq[j]
        
        if i > 0: new_sum += overlap[seq[i-1]][pj]
        if i < N-1: new_sum += overlap[pj][seq[i+1]]
        if j > 0: new_sum += overlap[seq[j-1]][pi]
        if j < N-1: new_sum += overlap[pi][seq[j+1]]
        
        if j == i + 1 or i == j + 1:
            # Middle edge is now (pi, pj) or (pj, pi) depending on positions
            # If j = i+1:
            # pos i has pj, pos j has pi.
            # Edge between i and j is (pj, pi).
            new_sum -= overlap[pj][pi]
            
        delta = new_sum - old_sum
        
        if delta > 0 or random.random() < pow(2.71828, -delta / T):
            seq[i], seq[j] = seq[j], seq[i]
            current_overlap += delta
            if current_overlap > best_overlap:
                best_overlap = current_overlap
                best_seq = seq[:]
                
        T *= cooling

    # Construct final string from best_seq
    # Start with first perm
    final_word = list(perms[best_seq[0]])
    for k in range(1, len(best_seq)):
        p_prev = perms[best_seq[k-1]]
        p_curr = perms[best_seq[k]]
        ov = overlap[best_seq[k-1]][best_seq[k]]
        # Append p_curr[ov:]
        final_word.extend(p_curr[ov:])
        
    return "".join(map(str, final_word))
