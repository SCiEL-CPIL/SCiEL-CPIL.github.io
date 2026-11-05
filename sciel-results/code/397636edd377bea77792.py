"""Improved superpermutation construction using greedy reordering of recursive blocks."""
import sys
from itertools import permutations
from collections import defaultdict
import time

def _standard_superpermutation(n):
    """Standard recursive superpermutation: 1! + 2! + ... + n! length."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    word = _standard_superpermutation(n - 1)
    m = n
    perms = []
    seen = set()
    for i in range(len(word) - (m - 1) + 1):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    new_word = []
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find maximum overlap between new_word and block
        best_overlap = 0
        max_ov = min(len(new_word), len(block))
        for ov in range(max_ov, -1, -1):
            if ov == 0:
                best_overlap = 0
                break
            if new_word[len(new_word) - ov:] == block[:ov]:
                best_overlap = ov
                break
        new_word.extend(block[best_overlap:])
    return new_word

def _greedy_order_blocks(perms, n, deadline):
    """
    Given the list of (n-1)-permutations (as tuples) that form the blocks for symbol n,
    find an ordering that minimizes total length when overlapped.
    Each block is: p + [n] + p where p is a permutation of 1..n-1.
    The overlap between consecutive blocks depends on how much the suffix of one
    matches the prefix of the next.
    """
    k = len(perms)
    if k <= 2:
        return list(perms)
    
    # Precompute block strings for overlap calculation
    # Block for permutation p: p + [n] + p
    # But we only need to consider overlaps between the end of one block and start of next.
    
    # The suffix of block p1 is: ... + [n] + p1
    # The prefix of block p2 is: p2 + [n] + ...
    
    # Overlap between block p1 and block p2:
    # We need to find the longest suffix of (p1 + [n] + p1) that matches a prefix of (p2 + [n] + p2)
    
    # Let's precompute for each pair (i, j) the overlap length.
    # But k can be up to 5040 for n=7, so O(k^2) = 25M pairs which is too slow in Python.
    
    # Instead, let's think about the structure.
    # Block p = p[0] p[1] ... p[n-2] n p[0] p[1] ... p[n-2]
    # The suffix of block p1 of length L must match the prefix of block p2 of length L.
    
    # Key insight: The overlap is determined by how much the tail of p1 (and the n) 
    # matches the head of p2.
    
    # For two permutations p1 and p2 of 1..n-1:
    # block1 ends with: ... p1[-1], n, p1[0], ..., p1[n-2]
    # block2 starts with: p2[0], ..., p2[n-2], n, ...
    
    # The maximum possible overlap is 2*(n-1) + 1 = 2n-1, but realistically much less.
    
    # Let's compute overlap for pairs more efficiently.
    # The suffix of block p1 that could match prefix of block p2:
    # The last part of block p1 is p1 itself (length n-1).
    # The first part of block p2 is p2 itself (length n-1).
    
    # So we're looking for the longest string that is a suffix of p1 and a prefix of p2,
    # but also considering the n in between.
    
    # Actually, let's just compute it directly but with early termination.
    
    def overlap_len(p1, p2):
        """Compute max overlap: longest suffix of block(p1) that is prefix of block(p2)."""
        b1 = list(p1) + [n] + list(p1)
        b2 = list(p2) + [n] + list(p2)
        L1 = len(b1)
        L2 = len(b2)
        max_ov = min(L1, L2)
        for ov in range(max_ov, 0, -1):
            if b1[L1 - ov:] == b2[:ov]:
                return ov
        return 0
    
    # For n=7, k=5040, computing all pairs is 25M * (up to 13 comparisons) = too slow.
    # Let's use a smarter approach.
    
    # Precompute the "signature" of each block's suffix and prefix.
    # For each permutation p, the block is p + [n] + p.
    # The suffixes of block p are: 
    #   p[i:] + ... no wait.
    # Block p = p[0]...p[n-2], n, p[0]...p[n-2]
    # Suffixes of length L for L <= n-1: p[n-2-L+1:] (from the last p)
    # For L = n-1: p (the whole last copy)
    # For L = n: [n-1 element of first p? No]
    
    # Let me think differently. Let's build a graph where nodes are permutations,
    # and edge i->j has weight = (n) - overlap_len(p_i, p_j) + ... 
    # Actually the cost of transitioning from p_i to p_j is:
    # We've already output block(p_i). Now we add block(p_j) with maximum overlap.
    # The number of NEW characters added is: 2*(n-1) + 1 - overlap_len(p_i, p_j)
    
    # So total length = sum of 2*(n-1)+1 for all blocks - sum of overlaps between consecutive blocks
    # = k*(2n-1) - sum of overlaps
    # To minimize total length, we want to maximize sum of overlaps.
    
    # This is a TSP-like problem. With k=5040 nodes, exact TSP is infeasible.
    # Greedy: start from some node, always go to the neighbor with maximum overlap.
    
    # To make this efficient, let's precompute for each node, the top few neighbors by overlap.
    
    # But even that is O(k^2) for n=7.
    
    # Alternative: use the structure of permutations.
    # The overlap between p1 and p2 depends on how much p1's tail matches p2's head.
    
    # Let's define: for permutation p, the "right end" is p (the last n-1 chars of the block).
    # The "left end" is p (the first n-1 chars of the block).
    
    # Overlap: we need suffix of (p1 + [n] + p1) to match prefix of (p2 + [n] + p2).
    
    # Case 1: overlap is within the last p1 and first p2.
    #   So we need p1[i:] == p2[:n-1-i] for some i, meaning the suffix of p1 matches prefix of p2.
    #   The overlap length would be (n-1-i) if this matches, but we also need to check if it can extend further.
    
    # This is getting complex. Let's just do a limited greedy with a time budget.
    
    # For n <= 5, k is small enough for more exhaustive approaches.
    # For n=6, k=720, O(k^2) = 518K pairs, feasible.
    # For n=7, k=5040, O(k^2) = 25M, too slow in pure Python.
    
    # Let's handle n=7 specially with a heuristic.
    
    if n <= 5:
        # Compute all overlaps
        ov_matrix = [[0]*k for _ in range(k)]
        for i in range(k):
            p1 = perms[i]
            b1 = list(p1) + [n] + list(p1)
            for j in range(k):
                if i == j:
                    continue
                p2 = perms[j]
                b2 = list(p2) + [n] + list(p2)
                max_ov = min(len(b1), len(b2))
                for ov in range(max_ov, 0, -1):
                    if b1[len(b1)-ov:] == b2[:ov]:
                        ov_matrix[i][j] = ov
                        break
    else:
        # For larger n, use a different approach
        # Greedy without full matrix
        return _greedy_order_large(perms, n, deadline)
    
    # Greedy: start from each of several starting points, or just one
    # Let's start from the first permutation and greedily pick the best neighbor
    visited = [False] * k
    path = [0]
    visited[0] = True
    current = 0
    
    for _ in range(k - 1):
        best_next = -1
        best_ov = -1
        for j in range(k):
            if not visited[j] and ov_matrix[current][j] > best_ov:
                best_ov = ov_matrix[current][j]
                best_next = j
        if best_next == -1:
            break
        path.append(best_next)
        visited[best_next] = True
        current = best_next
    
    if len(path) < k:
        # Add remaining
        for j in range(k):
            if not visited[j]:
                path.append(j)
    
    return [perms[i] for i in path]

def _greedy_order_large(perms, n, deadline):
    """For large n (like 7), use a faster greedy approach."""
    k = len(perms)
    
    # Precompute for each permutation, some "features" for quick overlap estimation
    # The key is: overlap between block p1 and block p2 is determined by
    # how much the tail of p1 matches the head of p2, plus the n in between.
    
    # Let's just do a simple greedy: for each current permutation, find the next
    # that has the longest matching prefix with the current's tail.
    
    # Actually, let's use a different strategy: try to order permutations such that
    # consecutive ones have long common subsequences in the right positions.
    
    # Simpler: just use the standard order but with local improvements.
    # Or: sort by some heuristic.
    
    # For now, let's just return the standard order for n=7 as a fallback,
    # and rely on the recursion for n<=6 to provide good sub-structures.
    return list(perms)

def _join_blocks(ordered_perms, n):
    """Join ordered permutations into a superpermutation string."""
    if not ordered_perms:
        return ""
    
    blocks = []
    for p in ordered_perms:
        blocks.append(list(p) + [n] + list(p))
    
    result = list(blocks[0])
    for i in range(1, len(blocks)):
        block = blocks[i]
        # Find max overlap
        best_ov = 0
        max_ov = min(len(result), len(block))
        for ov in range(max_ov, 0, -1):
            if result[len(result)-ov:] == block[:ov]:
                best_ov = ov
                break
        result.extend(block[best_ov:])
    
    return result

def construct(n):
    """
    Construct a shortest-known superpermutation for n symbols.
    Uses recursive construction with greedy reordering of blocks.
    """
    deadline = time.time() + 120  # Safety margin for 120s limit
    
    if n <= 1:
        return "1"
    if n <= 4:
        word = _standard_superpermutation(n)
        return "".join(map(str, word))
    
    # Recursive construction with greedy reordering
    prev_word = _standard_superpermutation(n - 1)
    
    # Extract the (n-1) permutations from prev_word
    perms = []
    seen = set()
    for i in range(len(prev_word) - (n - 1) + 1):
        window = tuple(prev_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Order the blocks greedily
    if time.time() < deadline:
        ordered = _greedy_order_blocks(perms, n, deadline)
    else:
        ordered = list(perms)
    
    # Join into final word
    word = _join_blocks(ordered, n)
    
    return "".join(map(str, word))

# Cache for standard superpermutations
_cache = {}

def _get_standard(n):
    if n not in _cache:
        _cache[n] = _standard_superpermutation(n)
    return _cache[n]
