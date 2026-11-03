"""Superpermutation constructor using de Bruijn graph Eulerian path approach."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque
from itertools import permutations

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _build_de_bruijn(n):
    """Build de Bruijn graph for (n-1)-permutations.
    Nodes are (n-1)-permutations of 1..n.
    Edge from u to v exists if v[1:] == u[:-1] + [x] for some x not in u.
    Actually: edge from u to v where v = (u[1], u[2], ..., u[n-2], x) for x not in u.
    """
    # All (n-1)-permutations of 1..n
    nodes = list(permutations(range(1, n + 1), n - 1))
    node_set = set(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # For each node, find outgoing edges
    # Edge from u to v: v = (u[1], ..., u[n-2], x) where x not in u
    # This means the last element of v is x, and v[0:n-2] = u[1:n-1]
    out_edges = defaultdict(list)
    in_edges = defaultdict(list)
    
    for u in nodes:
        # u is a tuple of length n-1
        # Possible next symbols: x not in u
        # v = (u[1], u[2], ..., u[n-2], x)
        for x in range(1, n + 1):
            if x not in u:
                v = u[1:] + (x,)
                out_edges[u].append(v)
                in_edges[v].append(u)
    
    return nodes, node_to_idx, out_edges, in_edges

def _eulerian_path_de_bruijn(n, time_limit):
    """Find an Eulerian path in the de Bruijn graph using Hierholzer's algorithm.
    Returns the sequence of symbols if successful, else None.
    """
    start_time = time.time()
    
    nodes, node_to_idx, out_edges, in_edges = _build_de_bruijn(n)
    num_nodes = len(nodes)
    
    # Check if Eulerian path exists:
    # For an Eulerian path (not circuit), exactly 0 or 2 nodes have odd degree,
    # and the start has out-in=1, end has in-out=1, all others balanced.
    # In our graph, each node has exactly 2 outgoing edges (for n>=3, since n-1 symbols
    # in the node, so 1 symbol not in node... wait, n symbols total, n-1 in node,
    # so 1 symbol not in node. So each node has exactly 1 outgoing edge!)
    # Wait: for (n-1)-permutations of n symbols, each node has n-1 distinct symbols.
    # The number of symbols not in the node is n - (n-1) = 1.
    # So each node has exactly 1 outgoing edge!
    # This means the graph is a set of cycles (each node has in-degree 1 and out-degree 1).
    # An Eulerian circuit exists in each connected component.
    # But we need a single path that covers all edges.
    
    # Actually, let me reconsider. The de Bruijn graph B(n, n-1) where nodes are
    # (n-1)-permutations (no repeated symbols) of {1,...,n}:
    # Each node u has exactly one symbol x not in u, so exactly one outgoing edge to v = u[1:] + (x,)
    # Similarly, each node has exactly one incoming edge.
    # So the graph is a disjoint union of cycles!
    # The number of edges = number of nodes = n! (since there are n! permutations of n symbols,
    # and each (n-1)-permutation is a prefix of exactly one n-permutation... no wait)
    # Number of (n-1)-permutations of n symbols: P(n, n-1) = n! / 1! = n!
    # So there are n! nodes, each with 1 in and 1 out edge.
    # The graph decomposes into cycles.
    
    # This is a known structure. The number of cycles in B(n, n-1) is (n-1)! for the
    # case where we consider all (n-1)-permutations. Actually, let me just find the cycles.
    
    # Find all cycles
    visited = set()
    cycles = []
    for node in nodes:
        if node not in visited:
            # Follow the cycle
            cycle = []
            curr = node
            while curr not in visited:
                visited.add(curr)
                cycle.append(curr)
                # Next node: the unique outgoing edge
                # Find x not in curr
                in_curr = set(curr)
                for x in range(1, n + 1):
                    if x not in in_curr:
                        curr = curr[1:] + (x,)
                        break
            cycles.append(cycle)
    
    # Now we need to merge these cycles into a single path.
    # The classic construction does this by inserting cycles.
    # A superpermutation corresponds to an ordering of the n! edges (which correspond
    # to the n! permutations) such that consecutive edges share a node.
    
    # Actually, let me think differently. Each edge in the de Bruijn graph corresponds
    # to an n-permutation: edge from u to v where v = u[1:] + (x,) corresponds to
    # the permutation u + (x,) (which is u followed by x). Wait, u is length n-1,
    # so u + (x,) is length n. That's the permutation.
    
    # So we have n! edges, each labeled by an n-permutation.
    # We need to order all n! edges such that consecutive edges are connected
    # (end of one = start of next), and the total length of the resulting word
    # is minimized.
    
    # If we can find a Hamiltonian cycle in the line graph (or equivalently,
    # an Eulerian circuit in the original graph), the word length would be n! + n - 1
    # (since we traverse n! edges, each adding 1 symbol, plus the initial n-1 symbols).
    # But the graph is a union of cycles, so an Eulerian circuit exists only if
    # the graph is connected. It's not connected in general.
    
    # The minimum superpermutation length is n! + (n-1)! + (n-2)! + ... + 1! - (n-1)
    # for the classic construction, which is 5913 - 6 = 5907 for n=7? No,
    # classic is 1!+2!+...+n! = 5913 for n=7.
    
    # Let me just use the classic construction and then try to optimize.
    # The de Bruijn approach with cycle merging is essentially what the classic
    # construction does.
    
    return None  # Fall back to classic

def _optimize_word(word, n, time_limit):
    """Try to shorten the word by removing redundant characters."""
    start_time = time.time()
    word = list(word)
    L = len(word)
    
    # Build set of all permutations for quick lookup
    all_perms = set(permutations(range(1, n + 1)))
    
    def get_windows(w):
        """Get set of all n-windows in w."""
        windows = set()
        for i in range(len(w) - n + 1):
            windows.add(tuple(w[i:i+n]))
        return windows
    
    def is_valid(w):
        """Check if w contains all permutations."""
        if len(w) < n:
            return False
        windows = set()
        for i in range(len(w) - n + 1):
            windows.add(tuple(w[i:i+n]))
            if len(windows) >= len(all_perms):
                break
        return all_perms.issubset(windows)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try removing each character from right to left
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            if L <= n:
                break
            
            # Quick check: if this position is part of a unique window, skip
            # (but we'll do the full check anyway for correctness)
            test_word = word[:i] + word[i+1:]
            if len(test_word) < n:
                continue
            
            if is_valid(test_word):
                word = test_word
                L = len(word)
                improved = True
                break  # Restart scan from right
    
    return word

def _two_opt(word, n, time_limit):
    """Try 2-opt style improvements: reverse segments."""
    start_time = time.time()
    word = list(word)
    L = len(word)
    
    all_perms = set(permutations(range(1, n + 1)))
    
    def is_valid(w):
        if len(w) < n:
            return False
        windows = set()
        for i in range(len(w) - n + 1):
            windows.add(tuple(w[i:i+n]))
            if len(windows) >= len(all_perms):
                break
        return all_perms.issubset(windows)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try reversing various segments
        for length in range(2, min(L // 2, 20)):
            if (time.time() - start_time) >= time_limit:
                break
            for start in range(0, L - length + 1, max(1, (L - length) // 50)):
                if (time.time() - start_time) >= time_limit:
                    break
                # Reverse word[start:start+length]
                seg = word[start:start+length]
                word[start:start+length] = seg[::-1]
                
                if is_valid(word):
                    # Check if this is shorter or same length (it's same length, so only useful
                    # if it enables further deletions)
                    # For now, just accept if valid
                    improved = True
                    break
                else:
                    # Revert
                    word[start:start+length] = seg
            if improved:
                break
    
    return word

def construct(n):
    start_time = time.time()
    total_limit = 140  # Safety margin for 150s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    elapsed = time.time() - start_time
    remaining = total_limit - elapsed
    
    if remaining > 5:
        # Try to remove redundant characters
        word = _optimize_word(word, n, remaining * 0.7)
    
    elapsed = time.time() - start_time
    remaining = total_limit - elapsed
    
    if remaining > 5:
        # Try 2-opt improvements
        word = _two_opt(word, n, remaining * 0.8)
    
    # Convert to string
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
