import sys
from itertools import permutations
from collections import defaultdict

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Build the graph:
    # Nodes: (n-1)-tuples of distinct symbols from {1..n}
    # Edge from u to v where v = u[1:] + (missing_symbol,)
    # Each edge corresponds to an n-permutation: u + (missing_symbol,)
    
    all_syms = tuple(range(1, n+1))
    all_syms_set = set(all_syms)
    
    def get_missing(node):
        # node is a tuple of n-1 distinct symbols
        # Return the one symbol not in node
        node_set = set(node)
        for s in all_syms:
            if s not in node_set:
                return s
        return None
    
    def get_next(node):
        s = get_missing(node)
        return node[1:] + (s,)
    
    def node_to_perm(node):
        s = get_missing(node)
        return node + (s,)
    
    # Find all cycles
    all_nodes = set(permutations(range(1, n+1), n-1))
    visited = set()
    cycles = []
    
    for node in all_nodes:
        if node not in visited:
            cycle = []
            cur = node
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                cur = get_next(cur)
            cycles.append(cycle)
    
    # For each cycle, compute the list of n-permutations
    cycle_perms = []
    for cycle in cycles:
        perms = [node_to_perm(node) for node in cycle]
        cycle_perms.append(perms)
    
    m = len(cycles)
    
    # Precompute overlaps between any suffix of cycle i and any prefix of cycle j
    # overlap[i][j] = max overlap between any node in cycle i and any node in cycle j
    # Actually, we need overlap between the end of the traversal of cycle i and start of cycle j
    # But since we can rotate cycles, we need to find the best entry/exit points
    
    # For each pair (i, j), compute the maximum overlap between any node in cycle i 
    # and any node in cycle j (suffix of node in i matching prefix of node in j)
    # This allows us to choose optimal entry/exit points for each cycle
    
    # Precompute for each cycle, for each node in it, the permutation it represents
    # and the next/prev nodes in the cycle
    
    # For efficiency, precompute suffixes and prefixes
    # For each node, store all its suffixes of length 1..n-1
    # For each node, store all its prefixes of length 1..n-1
    
    # Build a map from (suffix_tuple, cycle_idx, node_idx_in_cycle) to allow lookup
    # and from (prefix_tuple, cycle_idx, node_idx_in_cycle)
    
    suffix_map = defaultdict(list)  # suffix -> list of (cycle_idx, node_idx_in_cycle)
    prefix_map = defaultdict(list)  # prefix -> list of (cycle_idx, node_idx_in_cycle)
    
    for ci in range(m):
        for ni, node in enumerate(cycles[ci]):
            # Add all suffixes
            for k in range(1, n):
                suffix = node[-k:]
                suffix_map[suffix].append((ci, ni))
            # Add all prefixes
            for k in range(1, n):
                prefix = node[:k]
                prefix_map[prefix].append((ci, ni))
    
    # Compute best overlap for each pair of cycles (i, j)
    # best_overlap[i][j] = max k such that there exists a node in cycle i with suffix of length k
    # matching a node in cycle j with prefix of length k
    best_overlap = [[0]*m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            # Find max k
            max_ovl = 0
            for k in range(n-1, 0, -1):
                found = False
                # Check if any node in cycle i has a suffix of length k that matches
                # a prefix of length k in cycle j
                for suffix, locs in suffix_map.items():
                    if len(suffix) != k:
                        continue
                    # Check if this suffix appears in cycle i
                    in_i = any(c == i for c, _ in locs)
                    if not in_i:
                        continue
                    # Check if this suffix appears as a prefix in cycle j
                    prefix_locs = prefix_map.get(suffix, [])
                    in_j = any(c == j for c, _ in prefix_locs)
                    if in_j:
                        found = True
                        break
                if found:
                    max_ovl = k
                    break
            best_overlap[i][j] = max_ovl
    
    # Now use a greedy + 2-Opt approach to order the cycles
    # to maximize the sum of overlaps between consecutive cycles
    
    # Greedy: start with the cycle that has the best average outgoing overlap
    used = [False] * m
    order = []
    
    best_start = 0
    best_score = -1
    for i in range(m):
        score = sum(best_overlap[i][j] for j in range(m) if j != i)
        if score > best_score:
            best_score = score
            best_start = i
    
    order.append(best_start)
    used[best_start] = True
    current = best_start
    
    for _ in range(m - 1):
        best_next = -1
        best_ovl = -1
        for j in range(m):
            if not used[j]:
                if best_overlap[current][j] > best_ovl:
                    best_ovl = best_overlap[current][j]
                    best_next = j
        if best_next == -1:
            for j in range(m):
                if not used[j]:
                    best_next = j
                    break
        order.append(best_next)
        used[best_next] = True
        current = best_next
    
    # 2-Opt improvement
    def compute_order_score(ord):
        total = 0
        for i in range(len(ord) - 1):
            total += best_overlap[ord[i]][ord[i+1]]
        return total
    
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            for j in range(i + 1, len(order)):
                # Try reversing the segment from i to j
                new_order = order[:i] + order[i:j+1][::-1] + order[j+1:]
                if compute_order_score(new_order) > compute_order_score(order):
                    order = new_order
                    improved = True
    
    # Build the word
    # For each cycle in order, we need to choose the best entry/exit points
    # that maximize overlap with the previous cycle's exit and next cycle's entry
    
    # For simplicity, let's first just use the first node of each cycle as entry
    # and the last as exit, then refine
    
    # Actually, let's be smarter: for each cycle, given the previous cycle's exit node
    # and the next cycle's entry node, find the rotation that maximizes overlap
    
    # For now, let's just build the word using the cycles in order,
    # with the entry/exit chosen to maximize overlap with neighbors
    
    # For each cycle i in the order, we want to find a rotation (start index)
    # such that the overlap with the previous cycle's last node and the next cycle's first node is maximized
    
    # Let's precompute for each cycle and each possible start index, what the
    # "entry node" and "exit node" would be, and the overlaps with neighbors
    
    # This is getting complex. Let's use a simpler approach:
    # For each cycle, try all possible rotations and pick the one that
    # gives the best overlap with the previous cycle's exit
    
    word = []
    
    for oi, ci in enumerate(order):
        cycle = cycles[ci]
        L = len(cycle)
        
        if oi == 0:
            # First cycle: pick the rotation that gives best overlap with the next cycle
            next_ci = order[oi + 1] if oi + 1 < m else None
            best_start_idx = 0
            best_ovl = -1
            for si in range(L):
                exit_node = cycle[si - 1]  # The node before the start (cyclic)
                if next_ci is not None:
                    # Overlap between exit_node and any node in next cycle
                    # For simplicity, just use best_overlap
                    ovl = best_overlap[ci][next_ci]
                else:
                    ovl = 0
                if ovl > best_ovl:
                    best_ovl = ovl
                    best_start_idx = si
            
            # Start from best_start_idx
            for k in range(L):
                node = cycle[(best_start_idx + k) % L]
                perm = node_to_perm(node)
                if not word:
                    word.extend(perm)
                else:
                    max_ovl = 0
                    for k2 in range(1, min(n, len(word)+1)):
                        if k2 > len(word):
                            break
                        if tuple(word[-k2:]) == perm[:k2]:
                            max_ovl = k2
                    word.extend(perm[max_ovl:])
        else:
            # Non-first cycle: pick the rotation that gives best overlap with previous cycle's exit
            prev_ci = order[oi - 1]
            # The previous cycle's exit node is the last node we added
            # For simplicity, let's just use the first node of this cycle
            # and rely on the overlap computation during word building
            
            # Try all rotations and pick the one with best overlap with previous
            best_start_idx = 0
            best_ovl = -1
            for si in range(L):
                entry_node = cycle[si]
                # Overlap between previous cycle's exit and this entry
                # We don't easily know the previous exit, so use best_overlap
                ovl = best_overlap[prev_ci][ci]
                if ovl > best_ovl:
                    best_ovl = ovl
                    best_start_idx = si
            
            for k in range(L):
                node = cycle[(best_start_idx + k) % L]
                perm = node_to_perm(node)
                if not word:
                    word.extend(perm)
                else:
                    max_ovl = 0
                    for k2 in range(1, min(n, len(word)+1)):
                        if k2 > len(word):
                            break
                        if tuple(word[-k2:]) == perm[:k2]:
                            max_ovl = k2
                    word.extend(perm[max_ovl:])
    
    return "".join(map(str, word))
