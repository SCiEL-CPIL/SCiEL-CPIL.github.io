import itertools
import time
import sys

def _standard_superperm(n):
    """Generate standard superpermutation for n using recursive construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 2, 3, 1, 2, 1, 3, 2, 1]
    
    prev = _standard_superperm(n - 1)
    
    seen = set()
    order = []
    for i in range(len(prev) - (n - 1) + 1):
        window = tuple(prev[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    
    return new

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive subsequences."""
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in all_perms:
            found.add(p)
        if len(found) == len(all_perms):
            return True
    return len(found) == len(all_perms)

def _construct_from_order(order, n):
    """
    Construct superpermutation for n from the given order of (n-1)-permutations.
    Returns the word as a list of ints.
    """
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    return new

def _extract_order_from_word(word, n):
    """Extract the order of (n-1)-permutations from a superpermutation for n."""
    seen = set()
    order = []
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    return order

def _compute_overlap_matrix(perms, n):
    """
    Compute the overlap matrix K[i][j] = length of longest suffix of perms[i] 
    that is a prefix of perms[j].
    """
    num_perms = len(perms)
    K = [[0] * num_perms for _ in range(num_perms)]
    len_p = n - 1
    
    for i in range(num_perms):
        p_i = perms[i]
        for j in range(num_perms):
            if i == j:
                K[i][j] = len_p # Not used for i!=j in path sum, but good for completeness
                continue
            p_j = perms[j]
            # Max overlap is len_p - 1 since i != j
            # We check from max possible down to 0
            for ov in range(len_p - 1, -1, -1):
                if p_i[len_p - ov:] == p_j[:ov]:
                    K[i][j] = ov
                    break
    return K

def _total_overlap(order, K):
    """Calculate total overlap for a given order."""
    total = 0
    for i in range(len(order) - 1):
        total += K[order[i]][order[i+1]]
    return total

def _two_opt_order(order, K, deadline):
    """
    Improve the order using 2-opt local search to maximize total overlap.
    """
    N = len(order)
    if N <= 2:
        return order
    
    current_order = list(order)
    current_overlap = _total_overlap(current_order, K)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline:
                break
            for j in range(i + 2, N):
                # Calculate gain of reversing segment [i, j]
                # Edges removed:
                # 1. (current_order[i-1], current_order[i]) if i > 0
                # 2. (current_order[j], current_order[j+1]) if j < N-1
                # 3. Internal edges (k, k+1) for k in [i, j-1]
                
                # Edges added:
                # 1. (current_order[i-1], current_order[j]) if i > 0
                # 2. (current_order[i+1], current_order[i]) if i < N-1 (which is true since i < N-1 in loop)
                # 3. Internal edges (k+1, k) for k in [i, j-1]
                
                gain = 0
                
                # Left boundary
                if i > 0:
                    gain += K[current_order[i-1]][current_order[j]] - K[current_order[i-1]][current_order[i]]
                else:
                    # No left boundary cost/benefit in the sum definition
                    # The sum starts from index 0 to N-2
                    # If i=0, the first edge in the sum is (0,1).
                    # Reversing [0, j] changes the first edge to (j, j-1) ... wait.
                    # Original: P0, P1, ..., Pj
                    # New: Pj, Pj-1, ..., P0
                    # Original edges: (P0,P1), ..., (Pj-1,Pj)
                    # New edges: (Pj,Pj-1), ..., (P1,P0)
                    # So for i=0, we replace (P0,P1)...(Pj-1,Pj) with (Pj,Pj-1)...(P1,P0).
                    # And the right boundary (Pj, Pj+1) becomes (P0, Pj+1).
                    
                    # Let's re-evaluate the gain formula carefully.
                    # Sum S = sum_{k=0}^{N-2} K[P_k][P_{k+1}]
                    # Segment [i, j] reversed.
                    # P_i ... P_j becomes P_j ... P_i.
                    # 
                    # Edges affected:
                    # 1. Edge (P_{i-1}, P_i) -> (P_{i-1}, P_j)  [if i > 0]
                    # 2. Edge (P_j, P_{j+1}) -> (P_i, P_{j+1})  [if j < N-1]
                    # 3. Internal edges (P_k, P_{k+1}) for k=i..j-1 become (P_{k+1}, P_k)
                    
                    # If i=0:
                    # There is no (P_{-1}, P_0).
                    # The edge (P_0, P_1) is internal? No, it's the first edge.
                    # If we reverse [0, j], the new sequence starts with P_j.
                    # The new first edge is (P_j, P_{j-1}).
                    # The old first edge was (P_0, P_1).
                    # So we replace (P_0, P_1) ... (P_{j-1}, P_j) with (P_j, P_{j-1}) ... (P_1, P_0).
                    # And the right boundary edge (P_j, P_{j+1}) becomes (P_0, P_{j+1}).
                    
                    # Let's just calculate the sum of the affected edges directly to be safe and simple.
                    # N is small (720), so O(N) per pair is $720^3 \approx 3.7 \times 10^8$ which is too slow in Python.
                    # We must use the incremental gain.
                    
                    # For i=0:
                    # Gain = (Sum_{k=0}^{j-1} K[P_{k+1}][P_k]) - (Sum_{k=0}^{j-1} K[P_k][P_{k+1}])
                    #       + K[P_0][P_{j+1}] - K[P_j][P_{j+1}]  (if j < N-1)
                    
                    # For i > 0 and j < N-1:
                    # Gain = K[P_{i-1}][P_j] - K[P_{i-1}][P_i]
                    #       + K[P_i][P_{j+1}] - K[P_j][P_{j+1}]
                    #       + Sum_{k=i}^{j-1} (K[P_{k+1}][P_k] - K[P_k][P_{k+1}])
                    
                    # For i > 0 and j == N-1:
                    # Gain = K[P_{i-1}][P_j] - K[P_{i-1}][P_i]
                    #       + Sum_{k=i}^{j-1} (K[P_{k+1}][P_k] - K[P_k][P_{k+1}])
                    
                    # For i == 0 and j == N-1:
                    # Gain = Sum_{k=0}^{j-1} (K[P_{k+1}][P_k] - K[P_k][P_{k+1}])
                    
                    pass # Handled by specific cases below
        
                # Right boundary
                if j < N - 1:
                    gain += K[current_order[i]][current_order[j+1]] - K[current_order[j]][current_order[j+1]]
                else:
                    # No right boundary edge in sum
                    pass
                
                # Internal edges
                # Sum_{k=i}^{j-1} (K[P_{k+1}][P_k] - K[P_k][P_{k+1}])
                # We can precompute prefix sums of the "asymmetry" A[k] = K[P_k][P_{k+1}] - K[P_{k+1}][P_k]
                # Then sum is - (Sum_{k=i}^{j-1} A[k])
                
                # To make this O(1), we need prefix sums of A.
                # But A depends on the current order, which changes.
                # So we can't precompute globally.
                # However, N=720. $N^2/2$ pairs. $250k$ pairs.
                # If each pair takes $O(N)$ to sum, $250k \times 720 \approx 1.8 \times 10^8$.
                # This is borderline in Python (might take 10-20s per sweep).
                # With multiple sweeps, it might be too slow.
                
                # Let's try to optimize the inner loop.
                # We can break early if the gain is clearly negative?
                # Or just accept the cost if it's fast enough.
                
                # Let's implement the O(N) sum for now and see if it fits.
                # 150s limit.
                
                internal_gain = 0
                for k in range(i, j):
                    internal_gain += K[current_order[k+1]][current_order[k]] - K[current_order[k]][current_order[k+1]]
                
                gain += internal_gain
                
                if gain > 0:
                    # Reverse
                    current_order[i:j+1] = reversed(current_order[i:j+1])
                    current_overlap += gain
                    improved = True
    
    return current_order

def _delete_pass(word, n, deadline):
    """
    Character-level deletion pass.
    """
    S = list(word)
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
    
    best_len = len(S)
    
    while time.time() < deadline:
        made_removal = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            valid = True
            old_perms_to_dec = []
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms_to_dec.append(p)
            
            if not valid:
                continue
            
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n + 1)
            for k in range(start_new, end_new + 1):
                window_chars = []
                for idx in range(k, k+n):
                    s_idx = idx if idx < i else idx + 1
                    window_chars.append(S[s_idx])
                p_new = tuple(window_chars)
                counts[p_new] = counts.get(p_new, 0) + 1
            
            S.pop(i)
            best_len = len(S)
            made_removal = True
        
        if not made_removal:
            break
    
    return S

def construct(n):
    start_time = time.time()
    deadline = start_time + 140
    
    if n <= 5:
        word = _standard_superperm(n)
        if time.time() < deadline:
            word = _delete_pass(word, n, deadline)
        return "".join(map(str, word))
    
    perms = list(itertools.permutations(range(1, n)))
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Get initial order from standard construction
    base = _standard_superperm(n - 1)
    order_perms = _extract_order_from_word(base, n - 1)
    order_indices = [perm_to_idx[p] for p in order_perms]
    
    # Compute overlap matrix
    K = _compute_overlap_matrix(perms, n)
    
    # 2-opt optimization
    deadline_2opt = deadline - 20
    if time.time() < deadline_2opt:
        order_indices = _two_opt_order(order_indices, K, deadline_2opt)
    
    # Convert indices back to permutations
    order_perms = [perms[i] for i in order_indices]
    
    # Construct word
    word = _construct_from_order(order_perms, n)
    
    # Deletion pass
    if time.time() < deadline - 5:
        deadline_del = deadline - 5
        word = _delete_pass(word, n, deadline_del)
    
    if not _verify_superpermutation(word, n):
        word = _standard_superperm(n)
    
    return "".join(map(str, word))

if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}")
    
    for n in [6, 7]:
        start = time.time()
        w = construct(n)
        elapsed = time.time() - start
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}, time={elapsed:.2f}s")
