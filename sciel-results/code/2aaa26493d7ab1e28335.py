"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

# EVOLVE-BLOCK-START

def _reset_length(trans_a, trans_b):
    """Compute the exact shortest reset word length via BFS on subsets using bitmasks.
    Returns (length, word) or (None, None) if not synchronizing."""
    start_mask = (1 << N) - 1
    if start_mask == 0:
        return 0, ""
    
    # Precompute transitions for each letter as a list of bitmasks?
    # No, just compute on the fly.
    # visited: dict mapping mask -> (parent_mask, letter)
    # We need to reconstruct the word, so we store the path or just length if we only care about length.
    # The task says "returns a shortest word" in the docstring of the original, but construct() only returns the automaton.
    # The evaluator computes the word. So we just need the length to compare candidates.
    # However, the function signature implies it returns a word. I will return the length and a dummy word or reconstruct it.
    # For speed, I will just return the length. The caller in _construct_best only uses the length.
    
    visited = {start_mask: (-1, -1)} # mask -> (parent_mask, letter_idx)
    queue = deque([start_mask])
    dist = {start_mask: 0}
    
    while queue:
        mask = queue.popleft()
        d = dist[mask]
        for letter_idx, trans in enumerate((trans_a, trans_b)):
            new_mask = 0
            # Iterate through bits set in mask
            m = mask
            while m:
                lsb = m & -m
                state = lsb.bit_length() - 1
                dest = trans[state]
                new_mask |= (1 << dest)
                m ^= lsb
            
            if new_mask == 0:
                continue # Should not happen for complete automata
                
            # Check if singleton
            if new_mask & (new_mask - 1) == 0:
                return d + 1, "" # Word content not needed for selection
            
            if new_mask not in visited:
                visited[new_mask] = (mask, letter_idx)
                dist[new_mask] = d + 1
                queue.append(new_mask)
                
    return None, None


def _pinwheel(n, k):
    """
    Pinwheel automaton: states 0..n-1.
    Letter a: i -> (i+1) % n (cycle)
    Letter b: 0 -> 0, i -> i-1 for i=1..k-1, i -> 0 for i=k..n-1
    This is a known family with reset length n-1.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    for i in range(1, k):
        b[i] = i - 1
    # for i >= k, b[i] = 0 already
    return a, b


def _pinwheel_variant(n, k, offset=0):
    """
    Variant pinwheel with a shifted 'reset' region.
    a: cycle
    b: for states in a contiguous block of size k, they step towards one end;
       the rest map to a fixed point.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [0] * n
    # Block from (n-k) to n-1 wraps around? Let's do: states k..n-1 map down to k
    # Actually let's try: b[i] = i-1 for i in [1, k], b[i] = 0 for i > k
    # That's the same as _pinwheel.
    # Let's try a different structure:
    # a: cycle
    # b: 0->0, 1->0, 2->1, ..., k->k-1, k+1->0, ..., n-1->0
    # This is just pinwheel with k.
    # Alternative: "rank-2" style
    # Let's try: a: cycle, b: splits the cycle into two parts that collapse
    pass
    return a, b


def _rank2_cycle(n, k):
    """
    Rank-2 cycle construction.
    a: 0->1->2->...->n-1->0 (cycle)
    b: 0->0, 1->0, 2->1, 3->1, 4->2, 5->2, ..., pattern: b[i] = i//2 for i>=1, b[0]=0
    Actually let's use the standard rank-2:
    b[i] = i//2 for all i. Then b(0)=0, b(1)=0, b(2)=1, b(3)=1, etc.
    This has reset length roughly 2*(n-1) - something? Let's just compute.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [i // 2 for i in range(n)]
    return a, b


def _binary_tree_collapse(n):
    """
    a: cycle
    b: i -> i//2 (floor division), which collapses the state space.
    """
    a = [(i + 1) % n for i in range(n)]
    b = [i // 2 for i in range(n)]
    return a, b


def _improved_pinwheel(n):
    """
    Try pinwheel with k = n-1 (maximum).
    """
    return _pinwheel(n, n - 1)


def _construct_best():
    """
    Try several known constructions and return the one with the longest reset word.
    """
    candidates = []
    
    # 1. Standard Pinwheel (Cycle + Shift Down)
    # a: i -> i+1 mod N
    # b: 0->0, i->i-1 for i>0
    a = [(i + 1) % N for i in range(N)]
    b = [0 if i == 0 else i - 1 for i in range(N)]
    length, _ = _reset_length(a, b)
    if length is not None:
        candidates.append((length, a, b))
        
    # 2. Pinwheel variant: b maps high states to 0
    # This is often the "Crown" or similar
    for k in [N-1, N-2, N//2]:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        for i in range(1, k):
            b[i] = i - 1
        # b[i] = 0 for i >= k
        length, _ = _reset_length(a, b)
        if length is not None:
            candidates.append((length, a, b))

    # 3. Rank-2 / Binary Tree Collapse
    a = [(i + 1) % N for i in range(N)]
    b = [i // 2 for i in range(N)]
    length, _ = _reset_length(a, b)
    if length is not None:
        candidates.append((length, a, b))
        
    # 4. Another known good structure: "Two cycles"
    # a: swaps 0<->1, 2<->3, etc? No, a must be a permutation for high reset?
    # Let's stick to permutations for a to ensure synchronizability potential.
    
    if not candidates:
        # Fallback
        return [(i + 1) % N for i in range(N)], [0] * N
    
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_length, best_a, best_b = candidates[0]
    return best_a, best_b


def construct():
    # Directly construct the best known deterministic candidate
    # This avoids the slow local search and ensures fast execution
    best_a, best_b = _construct_best()
    return best_a, best_b

# EVOLVE-BLOCK-END
