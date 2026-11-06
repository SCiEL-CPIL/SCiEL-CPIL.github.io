import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # 1. Pre-compute all Q3 embeddings (clauses)
    # A Q3 is a 3-regular bipartite graph on 8 vertices.
    # We identify all distinct sets of 12 edges in K16 that form a Q3.
    # There are C(16, 8) = 12870 vertex subsets.
    # For each subset, there are 8! / 48 = 840 distinct Q3 subgraphs.
    # Total clauses: 12870 * 840 = 10,810,800.
    
    # To generate these efficiently, we can iterate over vertex subsets and generate
    # the 840 edge sets for each.
    # However, storing 10M clauses might be memory heavy (10M * 12 edges).
    # Instead, we can generate them on the fly or store them as bitmasks of edges.
    # Edge index is 0..120. A bitmask of 121 bits fits in a Python int (or 2 64-bit ints).
    
    # Let's precompute the 840 Q3 edge masks for the canonical vertices 0..7.
    # A Q3 on vertices 0..7 can be represented by a mapping f: {0,1}^3 -> {0,..,7}.
    # The edges are (f(x), f(x ^ e_i)) for i in 0,1,2 and x in {0,1}^3.
    
    # We need to generate all 840 distinct edge sets.
    # We can iterate over all 8! permutations of 0..7, assign them to the 8 cube vertices,
    # compute the 12 edges, and add the edge set (as a frozenset or bitmask) to a set.
    
    # Cube vertices:
    # 0: 000
    # 1: 001
    # 2: 010
    # 3: 011
    # 4: 100
    # 5: 101
    # 6: 110
    # 7: 111
    # Edges:
    # (0,1), (0,2), (0,4)
    # (1,3), (1,5)
    # (2,3), (2,6)
    # (3,7)
    # (4,5), (4,6)
    # (5,7)
    # (6,7)
    
    q3_edges_canonical = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7)
    ]
    
    # Generate all 840 unique edge masks for Q3 on vertices 0..7
    # Use a set of frozensets of edge indices (local 0..7)
    # Or better, use a set of bitmasks (128-bit int)
    
    q3_masks_local = set()
    for perm in permutations(range(8)):
        # perm[i] is the vertex label for cube vertex i
        mask = 0
        for u, v in q3_edges_canonical:
            a = perm[u]
            b = perm[v]
            if a > b:
                a, b = b, a
            # Local edge index: map (a,b) to a unique bit position
            # We can use a, b directly in a 64-bit mask? No, 8 vertices -> 28 edges.
            # Let's just use a frozenset of pairs for now, then convert to global.
            # Actually, for speed, let's precompute the 840 masks as bitsets of 28 local edges.
            # Local edge index:
            def local_edge_idx(u, v):
                if u > v: u, v = v, u
                # 0-1:0, 0-2:1, 0-3:2, 0-4:3, 0-5:4, 0-6:5, 0-7:6
                # 1-2:7, 1-3:8, 1-4:9, 1-5:10, 1-6:11, 1-7:12
                # 2-3:13, 2-4:14, 2-5:15, 2-6:16, 2-7:17
                # 3-4:18, 3-5:19, 3-6:20, 3-7:21
                # 4-5:22, 4-6:23, 4-7:24
                # 5-6:25, 5-7:26
                # 6-7:27
                if u == 0:
                    return v - 1
                if u == 1:
                    return 7 + (v - 2)
                if u == 2:
                    return 13 + (v - 3)
                if u == 3:
                    return 18 + (v - 4)
                if u == 4:
                    return 22 + (v - 5)
                if u == 5:
                    return 25 + (v - 6)
                if u == 6:
                    return 27
                return -1
            
            idx = local_edge_idx(a, b)
            mask |= (1 << idx)
        q3_masks_local.add(mask)
        
    # Verify count
    assert len(q3_masks_local) == 840, f"Expected 840, got {len(q3_masks_local)}"
    
    # Now, for each 8-vertex subset S of {0..15}, we can map the local mask to a global mask.
    # We will store the global masks in a list.
    # Global edge index is 0..120.
    # A global mask is a Python int (121 bits).
    
    all_q3_masks = []
    for S in combinations(range(n), 8):
        # Map local vertices 0..7 to global vertices S[0]..S[7]
        # We need to map local edge (u, v) to global edge (S[u], S[v])
        # And then to a global bit position.
        # We can precompute a mapping from local edge index (0..27) to global edge index (0..120) for this S.
        
        # Actually, it's faster to just iterate over the 840 local masks and convert them.
        # For a given S, local vertex i corresponds to global vertex S[i].
        # Local edge (u, v) corresponds to global edge (S[u], S[v]).
        # We need the global index of (S[u], S[v]).
        
        # Let's precompute a lookup table for this S: local_edge_idx -> global_edge_idx
        # There are 28 local edges.
        lookup = [0] * 28
        for i in range(8):
            for j in range(i + 1, 8):
                # Local edge (i, j)
                # Calculate local index
                # We can reuse the logic, but let's just compute global index directly
                u_g = S[i]
                v_g = S[j]
                if u_g > v_g:
                    u_g, v_g = v_g, u_g
                g_idx = edge_index[(u_g, v_g)]
                
                # We need to map this to the local bit position used in q3_masks_local
                # Let's just iterate over the bits in the local mask and set the corresponding global bit.
                pass
        
        # Alternative: For each local mask, iterate over its 12 set bits, map to global, and build global mask.
        # 12870 subsets * 840 masks * 12 bits = ~128M operations. This might be slow in Python.
        # 128M ops in Python ~ 10-20 seconds. This is acceptable for precomputation.
        
        for lmask in q3_masks_local:
            gmask = 0
            m = lmask
            while m:
                lidx = (m & -m).bit_length() - 1
                m ^= (1 << lidx)
                
                # Decode local edge index lidx to local vertices (u, v)
                # This is the inverse of local_edge_idx
                # We can precompute a list of local edges corresponding to indices 0..27
                # local_edges_list[lidx] = (u, v)
                
                # Let's precompute local_edges_list once
                # It's 28 pairs.
                
                # For efficiency, let's precompute the mapping for the current S
                # Actually, let's just precompute the global masks for ALL subsets in a vectorized way or optimized loop.
                
                # Let's try to optimize the inner loop.
                # Instead of bit iteration, we can store the 840 masks as lists of 12 local edge indices.
                # q3_local_edges[i] = [idx1, ..., idx12]
                
                pass
        
        # Let's refactor: Precompute q3_local_edges as a list of lists of indices.
        pass
        
    # Refactor Precomputation
    local_edges_list = []
    for i in range(8):
        for j in range(i + 1, 8):
            local_edges_list.append((i, j))
    # local_edges_list[k] corresponds to bit k in the mask.
    
    q3_local_edge_indices = []
    for lmask in q3_masks_local:
        inds = []
        m = lmask
        while m:
            lidx = (m & -m).bit_length() - 1
            m ^= (1 << lidx)
            inds.append(lidx)
        q3_local_edge_indices.append(inds)
        
    # Now generate global masks
    all_q3_masks = []
    for S in combinations(range(n), 8):
        # Precompute global index for each local edge index
        g_idx_map = [0] * 28
        for k, (u, v) in enumerate(local_edges_list):
            u_g = S[u]
            v_g = S[v]
            if u_g > v_g:
                u_g, v_g = v_g, u_g
            g_idx_map[k] = edge_index[(u_g, v_g)]
            
        for inds in q3_local_edge_indices:
            gmask = 0
            for k in inds:
                gmask |= (1 << g_idx_map[k])
            all_q3_masks.append(gmask)
            
    # Total masks: 10,810,800
    # Memory: 10M Python ints. Each int ~ 28-32 bytes. ~300MB.
    # This might be too much memory?
    # Let's check if we can store them more compactly or if memory is an issue.
    # If memory is an issue, we can sample clauses or use a different check.
    # But let's try this first.
    
    # Search Strategy: Max-SAT / Greedy Removal
    # Start with K16 (or dense random).
    # While there is a violated clause:
    #   Find a violated clause.
    #   Remove an edge from it.
    #   Which edge? The one that appears in the most OTHER violated clauses?
    #   Or just a random one?
    #   Or the one with highest degree?
    
    # To find a violated clause quickly:
    # We can maintain a counter for each edge: how many Q3s it is in that are currently "alive" (all edges present)?
    # Initially, all Q3s are alive.
    # When we remove an edge e, all Q3s containing e become dead.
    # We can maintain a list of Q3s for each edge.
    # edge_to_q3s[e] = list of indices of Q3s containing e.
    # q3_alive[i] = boolean.
    # violated_q3s = set of i where q3_alive[i] is True.
    
    # When we remove edge e:
    #   For each q3_idx in edge_to_q3s[e]:
    #       if q3_alive[q3_idx]:
    #           q3_alive[q3_idx] = False
    #           violated_q3s.remove(q3_idx)
    
    # This is efficient!
    # Precompute edge_to_q3s.
    # 10M Q3s * 12 edges = 120M entries.
    # This is a lot of memory (120M ints).
    # Maybe we can just pick a random Q3 from the list of all Q3s and check if it's alive?
    # If we keep a list of alive Q3s, and remove from it, it's fast.
    
    # Let's try this.
    
    edge_to_q3s = [[] for _ in range(num_edges)]
    for i, gmask in enumerate(all_q3_masks):
        m = gmask
        while m:
            e_idx = (m & -m).bit_length() - 1
            m ^= (1 << e_idx)
            edge_to_q3s[e_idx].append(i)
            
    # Now, the search.
    # Start with all edges present.
    # current_edges_mask = (1 << 121) - 1
    # alive_q3s = set(range(len(all_q3_masks)))
    
    # To avoid huge memory for alive_q3s, we can just iterate and check?
    # No, we need to find a violated Q3 quickly.
    # Let's keep a list of alive Q3s.
    # Initially, all are alive.
    # When we remove an edge, we mark Q3s as dead.
    # We can just keep a boolean array alive.
    # And a list of indices that are currently alive.
    # When we remove an edge, we iterate through its Q3s, and if alive, mark dead and remove from list.
    # Removing from a list is O(N).
    # Better: Use a set of alive Q3 indices.
    # alive_q3s = set(range(10M)) -> Too much memory/time to initialize.
    
    # Alternative:
    # Just pick a random Q3 index, check if it's violated (all 12 edges present).
    # If yes, remove an edge.
    # If no, pick another.
    # As the graph gets sparser, the probability of a random Q3 being violated drops.
    # But we can bias towards Q3s that are likely to be violated?
    
    # Let's try the "remove edge in most violated Q3s" heuristic.
    # We can maintain a count of violated Q3s per edge.
    # But updating counts is expensive.
    
    # Let's stick to:
    # 1. Start with K16.
    # 2. While True:
    #    a. Find a violated Q3 (random sampling or systematic).
    #    b. Remove an edge from it (heuristic: highest degree, or random).
    #    c. If no violated Q3 found after K samples, break.
    
    # To find a violated Q3:
    # We can iterate through a random permutation of Q3 indices.
    # Check if all 12 edges are in current_edges_mask.
    # Bitmask check is fast.
    
    random.seed(42)
    
    best_edges = []
    best_count = 0
    
    # We will do a few restarts with different initial graphs or removal orders.
    # But K16 is the densest start.
    # Maybe start with K16 and remove edges.
    
    # To improve, we can try to add edges back later?
    # No, if we start with K16 and remove, we end up with a subgraph of K16.
    # The optimal graph is a subgraph of K16.
    
    # Let's try to optimize the removal order.
    # Greedy: Remove edge that is in the most violated Q3s.
    # We can approximate this by:
    # 1. Find a violated Q3.
    # 2. Count how many other violated Q3s contain each edge of this Q3.
    #    (Sample other violated Q3s?)
    # 3. Remove the edge with the highest count.
    
    # This requires finding multiple violated Q3s.
    
    # Let's implement a simple version first:
    # Start with K16.
    # While violated Q3 exists:
    #   Find one (random search in list of all Q3s).
    #   Remove a random edge from it.
    
    # This is slow if we have to search many times.
    # Let's use the alive set idea but with a list of "candidate" Q3s.
    
    # Actually, let's try a different structural change:
    # Use a SAT solver? No, standard library only.
    # Use a greedy algorithm that builds the graph from scratch?
    # Start with empty graph, add edges if they don't create a Q3?
    # This would be slow (121 edges * 10M checks).
    
    # Let's stick to the deletion method but optimize the "find violated" step.
    # We can maintain a list of Q3s that are "potentially" violated.
    # Initially all.
    # When an edge is removed, remove all Q3s containing it from the list.
    # This is the "alive" list.
    # To avoid O(N) removal, we can use a "lazy deletion" approach.
    # Keep a list of all Q3s.
    # Keep a boolean array alive.
    # When looking for a violated Q3, iterate through the list, skip dead ones.
    # If we find a dead one, we can swap it with the last one and pop (if we maintain a list of alive indices).
    # But maintaining the list of alive indices requires updating it on every removal.
    
    # Let's try:
    # alive_indices = list(range(len(all_q3_masks)))
    # alive = [True] * len(all_q3_masks)
    
    # When removing edge e:
    #   for q3_idx in edge_to_q3s[e]:
    #       if alive[q3_idx]:
    #           alive[q3_idx] = False
    #           # Remove q3_idx from alive_indices
    #           # Find its position? O(N).
    #           # Better: just mark as dead, and when scanning, skip dead.
    #           # But the list will fill up with dead entries.
    #           # We can periodically rebuild the list.
    
    # Let's just scan the list of all Q3s.
    # 10M Q3s.
    # Check: (current_mask & q3_mask) == q3_mask.
    # This is very fast in Python if we use integers.
    # 10M checks ~ 0.1-0.5 seconds.
    # If we do this 100 times, it's 10-50 seconds.
    # This is feasible!
    
    # Strategy:
    # 1. current_mask = (1 << 121) - 1
    # 2. Loop:
    #    a. Find a violated Q3 by scanning all_q3_masks.
    #    b. If none found, break.
    #    c. Remove an edge from it.
    #       Heuristic: Remove the edge that is in the most violated Q3s?
    #       We can estimate this by checking a sample of other violated Q3s?
    #       Or just remove the edge with the highest degree in the current graph?
    #       Or random?
    
    # Let's try removing the edge with the highest degree first.
    
    def get_degree(e_idx, current_mask):
        # e_idx is index in all_edges
        # We need to count neighbors.
        # This is O(N) if we iterate over all edges.
        # Better to maintain a degree array.
        pass
    
    # Let's maintain a degree array.
    
    best_mask = 0
    best_count = 0
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        current_mask = (1 << num_edges) - 1
        degrees = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                e_idx = edge_index[(i, j)]
                # current_mask has all bits set initially
                degrees[i] += 1
                degrees[j] += 1
                
        # Actually, degrees are just 15 for all vertices in K16.
        # We need to update them as we remove edges.
        
        # Let's re-initialize degrees properly.
        degrees = [15] * n
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Find violated Q3
            violated_idx = -1
            for i, qmask in enumerate(all_q3_masks):
                if (current_mask & qmask) == qmask:
                    violated_idx = i
                    break
                    
            if violated_idx == -1:
                break
                
            # Get the edges of this Q3
            qmask = all_q3_masks[violated_idx]
            q_edges = []
            m = qmask
            while m:
                e_idx = (m & -m).bit_length() - 1
                m ^= (1 << e_idx)
                q_edges.append(e_idx)
                
            # Choose edge to remove
            # Heuristic: Max degree endpoint
            best_e_idx = -1
            best_score = -1
            for e_idx in q_edges:
                u, v = all_edges[e_idx]
                score = degrees[u] + degrees[v]
                if score > best_score:
                    best_score = score
                    best_e_idx = e_idx
                    
            # Remove it
            u, v = all_edges[best_e_idx]
            current_mask &= ~(1 << best_e_idx)
            degrees[u] -= 1
            degrees[v] -= 1
            
        # After loop, we have a Q3-free graph.
        # Try to add edges back?
        # We can do a hill climbing step: try to add any missing edge if it doesn't create a Q3.
        
        missing_edges = []
        for e_idx in range(num_edges):
            if not (current_mask & (1 << e_idx)):
                missing_edges.append(e_idx)
                
        random.shuffle(missing_edges)
        
        for e_idx in missing_edges:
            if time.time() > deadline - 5:
                break
            u, v = all_edges[e_idx]
            # Check if adding this edge creates a Q3
            # A Q3 is created if there is a Q3 that contains this edge and all other 11 edges are already present.
            # We can check all Q3s containing this edge.
            # edge_to_q3s[e_idx] has the list.
            creates_q3 = False
            for q3_idx in edge_to_q3s[e_idx]:
                qmask = all_q3_masks[q3_idx]
                # Check if (current_mask | (1 << e_idx)) contains qmask
                # Equivalent to: (current_mask & (qmask ^ (1 << e_idx))) == (qmask ^ (1 << e_idx))
                # i.e., all other edges are present.
                if (current_mask & (qmask ^ (1 << e_idx))) == (qmask ^ (1 << e_idx)):
                    creates_q3 = True
                    break
                    
            if not creates_q3:
                current_mask |= (1 << e_idx)
                degrees[u] += 1
                degrees[v] += 1
                
        # Update best
        count = bin(current_mask).count('1')
        if count > best_count:
            best_count = count
            best_mask = current_mask
            
    # Convert best_mask to list of edges
    result_edges = []
    for e_idx in range(num_edges):
        if best_mask & (1 << e_idx):
            result_edges.append(list(all_edges[e_idx]))
            
    return result_edges

# Helper for permutations
def permutations(iterable, r=None):
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = list(range(n))
    cycles = list(range(n, n - r, -1))
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in range(r - 1, -1, -1):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return
        cycles[:] = cycles[1:]
