"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # De Bruijn graph approach:
    # Nodes are (n-1)-tuples of distinct symbols from 1..n
    # Edge from node u to node v if there exists a symbol s such that
    # u + (s,) has all distinct symbols and v = u[1:] + (s,)
    # Each edge corresponds to an n-permutation.
    
    # We want to find a walk that uses every edge at least once, minimizing total edges.
    # This is the directed Chinese Postman Problem on a graph where each node has
    # out-degree 1 (since for a given (n-1)-tuple of distinct symbols, there's exactly
    # one symbol not in it).
    
    # Since out-degree = 1, the graph is a union of cycles.
    # We need to traverse all cycles. The minimal walk visits each edge exactly once
    # if we can connect the cycles, but since we can restart, the total length is
    # sum of cycle lengths + (number of cycles - 1) * 0 (we just concatenate).
    
    # Actually, the word length = n + (number of edges used - 1)
    # If we use each edge exactly once, length = n + n! - 1
    # But we might need to repeat some edges to connect cycles.
    
    # Let's find all cycles and then find the optimal way to traverse them.
    
    # For n=7, n! = 5040, (n-1)! = 720
    # Number of nodes in our graph: P(n, n-1) = n! / 1! ... wait
    
    # Let me reconsider. A node is an (n-1)-tuple of distinct symbols from {1,...,n}.
    # Number of such nodes = n * (n-1) * ... * 2 = n! / 1! = n!
    # Wait: P(n, n-1) = n! / (n-(n-1))! = n! / 1! = n!
    
    # So there are n! nodes and n! edges (one per n-permutation).
    # Each node has out-degree 1, so the graph is a permutation (collection of cycles).
    
    # Find all cycles:
    visited = set()
    cycles = []
    
    # Generate all (n-1)-tuples of distinct symbols from 1..n
    # There are n! of them, which is 5040 for n=7. Manageable.
    
    def get_next_node(node):
        # node is a tuple of n-1 distinct symbols
        # Find the symbol not in node
        all_syms = set(range(1, n+1))
        node_set = set(node)
        missing = (all_syms - node_set).pop()
        # Next node is node[1:] + (missing,)
        return node[1:] + (missing,)
    
    # We need to enumerate all nodes. Let's use permutations of n symbols taken n-1 at a time.
    # But that's n! nodes. For n=7, that's 5040. OK.
    
    all_nodes = set()
    for perm in permutations(range(1, n+1), n-1):
        all_nodes.add(perm)
    
    # Find cycles
    for node in all_nodes:
        if node not in visited:
            cycle = []
            cur = node
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = get_next_node(cur)
            cycles.append(cycle)
    
    # Now we have all cycles. To form a superpermutation, we need to traverse all edges.
    # Each cycle of length L contributes L edges.
    # The total number of edges is n! = 5040.
    # The word length if we traverse each edge exactly once in some order = n + n! - 1 = 5046?
    # No, that's not right either.
    
    # Let me think differently. A superpermutation is a sequence of n-permutations p1, p2, ..., pk
    # such that pi[1:] == p_{i+1}[:-1] for all i, and every n-permutation appears.
    # The word is p1[0] + p2[0] + ... + pk[0] + pk[1:] ... no.
    # The word is p1[0], p1[1], ..., p1[n-1], p2[n-1], p3[n-1], ..., pk[n-1]
    # Wait, since pi[1:] == p_{i+1}[:-1], we have:
    # word = p1[0], p1[1], ..., p1[n-1], p2[n-1], p3[n-1], ..., pk[n-1]
    # Length = n + (k-1)
    
    # So we want to minimize k, the number of permutations in the sequence.
    # The constraint is that consecutive permutations overlap in n-1 positions.
    
    # This is equivalent to finding a path in the graph (nodes = n-permutations,
    # edge from p to q if p[1:] == q[:-1]) that visits all nodes.
    # Each node has out-degree 1, so it's a collection of cycles.
    # We need to visit all n! nodes.
    
    # The minimal k is at least n! (we must visit each node at least once).
    # Can we achieve k = n!? That would require a single cycle containing all nodes,
    # i.e., the graph is one big cycle. But generally it's not.
    
    # For the graph where nodes are n-permutations and edge p->q if p[1:]==q[:-1]:
    # This is the same as the (n-1)-De Bruijn graph on permutations.
    # The number of cycles and their lengths determine the minimal k.
    
    # Actually, I realize the graph I should be using has nodes = (n-1)-permutations
    # (720 for n=7), not n-permutations (5040). Let me redo.
    
    # Redefine:
    # Nodes: all (n-1)-tuples of distinct symbols from {1,...,n}. Count = n! = 5040 for n=7.
    # Wait, P(7,6) = 7! = 5040. Yes.
    # Edge from u to v if v = u[1:] + (s,) where s is the unique symbol not in u.
    # Each edge is labeled by the n-permutation u + (s,).
    # There are n! edges, one per n-permutation.
    # Out-degree of each node = 1.
    
    # So the graph has 5040 nodes and 5040 edges, forming a permutation (cycles).
    # To cover all edges (all n-permutations), we need a walk that uses each edge at least once.
    # Since out-degree=1, any walk is deterministic once we pick the start.
    # The walk will eventually cycle. To cover all edges, we need to start at a node
    # and follow the unique outgoing edges until we've covered all edges.
    # But since the graph is a union of cycles, a walk starting at any node will
    # only cover the edges in its cycle.
    
    # So we need multiple walks (one per cycle), and we concatenate them.
    # The total word length = sum over cycles of (cycle_length + n - 1) 
    # minus overlaps between concatenated segments.
    
    # If we have cycles C1, C2, ..., Cm with lengths L1, L2, ..., Lm,
    # and we concatenate them, the total number of permutations used is sum(Li).
    # The word length = n + sum(Li) - 1 = n + n! - 1.
    # But we can overlap the end of one cycle with the start of the next if possible.
    
    # The end of a cycle is the last permutation in that cycle.
    # The start of the next cycle is the first permutation.
    # They can overlap if end[1:] == start[:-1], i.e., the last n-1 symbols of the
    # last permutation in cycle i match the first n-1 symbols of the first permutation
    # in cycle i+1.
    
    # Let's find all cycles and then find the best ordering and overlaps.
    
    # For n=7, 5040 nodes. Let's find the cycles.
    visited = set()
    cycles = []
    
    def get_next(node):
        s = (set(range(1, n+1)) - set(node)).pop()
        return node[1:] + (s,)
    
    all_nodes = set(permutations(range(1, n+1), n-1))
    
    for node in all_nodes:
        if node not in visited:
            cycle = []
            cur = node
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = get_next(cur)
            cycles.append(cycle)
    
    # Now, each cycle is a list of nodes. The edge from node i to node i+1 in the cycle
    # corresponds to the n-permutation node_i + (missing_symbol,).
    # The "permutation" at position i in the cycle is cycle[i] + (missing,).
    
    # Let's compute the actual n-permutations for each cycle.
    def node_to_perm(node):
        s = (set(range(1, n+1)) - set(node)).pop()
        return node + (s,)
    
    # For each cycle, the sequence of n-permutations is:
    # perm_i = cycle[i] + (s_i,) where s_i is the missing symbol for cycle[i]
    # And perm_i[1:] == perm_{i+1}[:-1] by construction.
    
    # Now we need to order the cycles and find overlaps.
    # The overlap between cycle A (ending with perm_A_last) and cycle B (starting with perm_B_first)
    # is the maximum k such that perm_A_last[n-k:] == perm_B_first[:k].
    # Since perm_A_last[1:] == perm_A_next[:-1] (but perm_A_next is in the same cycle),
    # the overlap between different cycles depends on the actual values.
    
    # For simplicity, let's just concatenate all cycles with minimal overlap.
    # The overlap between two n-permutations p and q is the largest k in [1, n-1]
    # such that p[n-k:] == q[:k].
    
    def overlap(p, q):
        max_k = 0
        for k in range(1, n):
            if p[n-k:] == q[:k]:
                max_k = k
        return max_k
    
    # Build the list of all permutations in cycle order.
    all_perms = []
    for cycle in cycles:
        for node in cycle:
            all_perms.append(node_to_perm(node))
    
    # Now find the best ordering of cycles to maximize overlap.
    # This is a TSP-like problem. For n=7, number of cycles might be small.
    # Let's check how many cycles there are.
    
    # Actually, let's just try a greedy approach: at each step, pick the next cycle
    # that has the maximum overlap with the current end.
    
    # First, let's compute the start and end permutations of each cycle.
    cycle_starts = []
    cycle_ends = []
    for cycle in cycles:
        start_perm = node_to_perm(cycle[0])
        end_perm = node_to_perm(cycle[-1])
        cycle_starts.append(start_perm)
        cycle_ends.append(end_perm)
    
    # Greedy: start with the first cycle, then repeatedly pick the cycle with max overlap.
    used = [False] * len(cycles)
    order = [0]
    used[0] = True
    current_end = cycle_ends[0]
    
    for _ in range(len(cycles) - 1):
        best_idx = -1
        best_ovl = 0
        for i in range(len(cycles)):
            if not used[i]:
                ovl = overlap(current_end, cycle_starts[i])
                if ovl > best_ovl:
                    best_ovl = ovl
                    best_idx = i
        if best_idx == -1:
            # Pick any unused
            for i in range(len(cycles)):
                if not used[i]:
                    best_idx = i
                    break
        order.append(best_idx)
        used[best_idx] = True
        current_end = cycle_ends[best_idx]
    
    # Now build the word.
    word = []
    first = True
    for ci in order:
        cycle = cycles[ci]
        for node in cycle:
            perm = node_to_perm(node)
            if first:
                word.extend(perm)
                first = False
            else:
                # Find overlap with current word end
                # The current word ends with the previous permutation's symbols.
                # We need to find how much of perm overlaps with the end of word.
                # Since consecutive permutations in the same cycle overlap by n-1,
                # and between cycles we computed the overlap.
                # Let's just find the max overlap between perm and the tail of word.
                max_ovl = 0
                for k in range(1, min(n, len(word))):
                    if tuple(word[-k:]) == perm[:k]:
                        max_ovl = k
                word.extend(perm[max_ovl:])
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
