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

def _compute_length_from_order(order, n):
    """
    Compute the exact length of the superpermutation built from the given order.
    Uses incremental string building with overlap detection.
    """
    total = 0
    current = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(current), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or current[len(current) - ov:] == block[:ov]:
                overlap = ov
                break
        total += len(block) - overlap
        current.extend(block[overlap:])
        # Keep current bounded for speed - we only need the tail
        if len(current) > 2 * n + 10:
            current = current[-(2 * n + 10):]
    return total

def _greedy_nearest_neighbor_order(perms, n):
    """
    Use a Greedy Nearest Neighbor heuristic to find a good ordering of (n-1)-permutations.
    The cost is the negative overlap (we want to maximize overlap).
    """
    num_perms = len(perms)
    if num_perms <= 1:
        return perms
    
    # Map permutation to index
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix
    # overlap[i][j] = length of longest suffix of perms[i] that is a prefix of perms[j]
    # Note: The block is P + [n] + P. 
    # The end of block i is P_i. The start of block j is P_j.
    # We need the longest suffix of P_i that matches a prefix of P_j.
    # Let k be the length of this match.
    # The overlap in the full construction is k.
    
    # Optimization: We only care about suffixes of P_i and prefixes of P_j.
    # P_i and P_j have length n-1.
    # Max possible overlap is n-2 (since the last char of P_i is part of the block end, 
    # but actually the block ends with P_i. The next block starts with P_j.
    # If P_i = a b c, P_j = b c d, overlap is 2.
    # If P_i = a b c, P_j = c a b, overlap is 1.
    # If P_i = a b c, P_j = d e f, overlap is 0.
    
    # We can precompute prefix hashes or just use string matching since n is small.
    # For n=7, n-1=6. 720 perms. 720*720 = 518k checks.
    # Each check is O(n^2) worst case, so 518k * 36 ~ 18M ops. Fast enough.
    
    overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        p_i = perms[i]
        # Suffixes of p_i
        # We want to match prefix of p_j
        for j in range(num_perms):
            if i == j:
                continue
            p_j = perms[j]
            # Try overlaps from max possible down to 0
            # Max overlap is len(p_i) - 1? No, if p_i == p_j, overlap is len(p_i).
            # But i != j, so p_i != p_j.
            # Max overlap is len(p_i) - 1 = n-2.
            # Actually, if p_i ends with X and p_j starts with X, and X is a permutation of subset?
            # No, p_i and p_j are permutations of 1..n-1.
            # If suffix of p_i is prefix of p_j, they must share the same elements in that segment.
            # Since they are permutations, if they share a segment of length k, that segment must be identical.
            
            max_ov = min(len(p_i), len(p_j)) - 1 # n-2
            for ov in range(max_ov, -1, -1):
                if p_i[len(p_i)-ov:] == p_j[:ov]:
                    overlap_matrix[i][j] = ov
                    break
    
    # Greedy TSP
    visited = [False] * num_perms
    order = []
    
    # Start with the permutation that has the highest average overlap? Or just 0.
    # Let's start with index 0.
    current = 0
    visited[0] = True
    order.append(perms[0])
    
    for _ in range(num_perms - 1):
        best_next = -1
        best_overlap = -1
        
        for next_idx in range(num_perms):
            if not visited[next_idx]:
                ov = overlap_matrix[current][next_idx]
                if ov > best_overlap:
                    best_overlap = ov
                    best_next = next_idx
        
        if best_next == -1:
            break
            
        visited[best_next] = True
        order.append(perms[best_next])
        current = best_next
        
    return order

def _delete_pass(word, n, deadline):
    """
    Character-level deletion pass. Try to remove characters that are redundant.
    A character can be removed if no permutation window that is unique is destroyed.
    """
    S = list(word)
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
    
    best_len = len(S)
    
    while time.time() < deadline:
        made_removal = False
        # Scan from right to left
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Old windows that include position i: indices i-n+1 to i
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
            
            # Perform removal
            for p in old_perms_to_dec:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            # New windows formed by straddling the gap
            # New string: S[:i] + S[i+1:]
            # Windows that straddle: indices k where k < i and k+n-1 >= i
            # i.e., k in [i-n+1, i-1] in new string coordinates
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 2 - n + 1)  # new len is len(S)-1
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
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # For n <= 5, return standard (optimal for n<=3)
    if n <= 5:
        word = _standard_superperm(n)
        # Apply deletion pass for potential improvement
        if time.time() < deadline:
            word = _delete_pass(word, n, deadline)
        return "".join(map(str, word))
    
    # For n >= 6: 
    # 1. Generate all (n-1)-permutations
    perms = list(itertools.permutations(range(1, n)))
    
    # 2. Use Greedy Nearest Neighbor to find a good order
    # This is a structural change from the recursive order extraction.
    # It directly optimizes the transition costs.
    if time.time() < deadline - 20:
        deadline_greedy = deadline - 20
        # We can try multiple random starts if time permits, but let's stick to one deterministic start first for stability.
        # Or we can pick the start with highest average overlap.
        # For now, start with the first permutation.
        order = _greedy_nearest_neighbor_order(perms, n)
    else:
        # Fallback to standard order if no time
        base = _standard_superperm(n - 1)
        order = _extract_order_from_word(base, n - 1)
        
    # 3. Construct the final superpermutation
    word = _construct_from_order(order, n)
    
    # 4. Character-level deletion pass
    if time.time() < deadline - 5:
        deadline_del = deadline - 5
        word = _delete_pass(word, n, deadline_del)
    
    # Verify
    if not _verify_superpermutation(word, n):
        # Fallback to standard
        word = _standard_superperm(n)
    
    return "".join(map(str, word))

# Quick self-test
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
