import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    n = 16
    
    # Q3 structure: 8 vertices (0-7), 12 edges
    # Q3 is the 3-dimensional hypercube
    Q3_EDGES = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                Q3_EDGES.append((i, j))
    
    Q3_ADJ = [set() for _ in range(8)]
    for a, b in Q3_EDGES:
        Q3_ADJ[a].add(b)
        Q3_ADJ[b].add(a)
    
    def find_q3(adj_bits, n=16):
        """
        Check if graph (adjacency as list of bitmasks) contains Q3 subgraph.
        Returns True if Q3 found.
        """
        assignment = [-1] * 8
        used_mask = 0
        
        def backtrack(idx):
            nonlocal used_mask
            if idx == 8:
                return True
            qv = idx
            # Build candidate bitmask: unused vertices adjacent to all assigned Q3-neighbors of qv
            candidates = ((1 << n) - 1) & ~used_mask
            for qn in Q3_ADJ[qv]:
                if assignment[qn] != -1:
                    candidates &= adj_bits[assignment[qn]]
            if candidates == 0:
                return False
            
            while candidates:
                lb = candidates & (-candidates)
                gv = lb.bit_length() - 1
                candidates ^= lb
                
                assignment[qv] = gv
                old_used = used_mask
                used_mask |= lb
                
                if backtrack(idx + 1):
                    return True
                
                used_mask = old_used
                assignment[qv] = -1
            
            return False
        
        for gv in range(n):
            assignment[0] = gv
            old_used = used_mask
            used_mask = (1 << gv)
            if backtrack(1):
                used_mask = old_used
                return True
            assignment[0] = -1
            used_mask = old_used
        
        return False
    
    def count_edges(bits, n=16):
        total = 0
        for i in range(n):
            total += bin(bits[i]).count('1')
        return total // 2
    
    def bits_to_edges(bits, n=16):
        edges = []
        for i in range(n):
            mask = bits[i]
            while mask:
                lb = mask & (-mask)
                j = lb.bit_length() - 1
                mask ^= lb
                if i < j:
                    edges.append([i, j])
        return edges
    
    # Precompute all possible Q3 embeddings (assignments of 8 Q3 vertices to 16 graph vertices)
    # There are C(16,8) * 8! = 12870 * 40320 = 518,918,400 total, but we can prune heavily
    # Instead, we'll use a different approach: enumerate all 8-subsets and check if they form Q3
    
    all_8sets = list(itertools.combinations(range(n), 8))
    
    def check_8set_is_q3(adj_bits, vertices):
        """Check if the induced subgraph on 'vertices' is isomorphic to Q3."""
        # Try all 8! = 40320 permutations
        # But we can prune: Q3 has specific degree sequence (all degree 3)
        # First check: each vertex in the induced subgraph must have degree 3
        for v in vertices:
            deg = 0
            for u in vertices:
                if u != v and (adj_bits[v] & (1 << u)):
                    deg += 1
            if deg != 3:
                return False
        
        # Now try to find a Q3 isomorphism
        # Use backtracking with pruning
        vlist = list(vertices)
        assignment = [-1] * 8
        used = set()
        
        def bt(idx):
            if idx == 8:
                return True
            qv = idx
            # Candidates: vertices in vlist not yet used, adjacent to all assigned Q3-neighbors
            candidates = []
            for gv in vlist:
                if gv in used:
                    continue
                valid = True
                for qn in Q3_ADJ[qv]:
                    if assignment[qn] != -1:
                        if not (adj_bits[gv] & (1 << assignment[qn])):
                            valid = False
                            break
                if valid:
                    candidates.append(gv)
            
            for gv in candidates:
                assignment[qv] = gv
                used.add(gv)
                if bt(idx + 1):
                    return True
                used.remove(gv)
                assignment[qv] = -1
            return False
        
        return bt(0)
    
    # Better approach: precompute which 8-sets CAN form Q3 (structural check)
    # Then for each such 8-set, the constraint is that the induced subgraph must NOT be Q3
    # This means at least one edge of the Q3 must be missing
    
    # Let's use a different strategy: 
    # Start with a dense graph and use local search to remove edges that create Q3s
    # Then try to add edges back
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    best_bits = None
    best_count = 0
    
    rng = random.Random(42)
    
    # Strategy: Start with complete graph, remove edges to break all Q3s
    # Then try to add edges back while maintaining Q3-free property
    
    # Phase 1: Find a Q3-free subgraph by removing edges
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            bits[i] |= (1 << j)
            bits[j] |= (1 << i)
    
    # Remove edges until Q3-free
    # Use greedy: repeatedly find a Q3 and remove an edge from it
    max_removals = 500
    for _ in range(max_removals):
        if time.time() > deadline - 30.0:
            break
        if not find_q3(bits, n):
            break
        # Find a Q3 and remove a random edge from it
        # For simplicity, try removing random edges
        removed = False
        for _attempt in range(100):
            u, v = rng.choice(all_edges)
            if bits[u] & (1 << v):
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
                if not find_q3(bits, n):
                    removed = True
                    break
                else:
                    bits[u] |= (1 << v)
                    bits[v] |= (1 << u)
        if not removed:
            # Force remove an edge
            u, v = all_edges[rng.randint(0, len(all_edges) - 1)]
            if bits[u] & (1 << v):
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
    
    if not find_q3(bits, n):
        cur_count = count_edges(bits, n)
        if cur_count > best_count:
            best_count = cur_count
            best_bits = list(bits)
    
    # Phase 2: Local search - try to add edges back
    missing_edges = [(i, j) for i, j in all_edges if not (bits[i] & (1 << j))]
    
    # Simulated annealing
    temp = 1.0
    cooling = 0.9995
    min_temp = 0.05
    step_limit = 100000
    
    for step in range(step_limit):
        if time.time() > deadline - 10.0:
            break
        if temp < min_temp:
            break
        
        # Try to add a random missing edge
        if missing_edges and rng.random() < 0.7:
            idx = rng.randint(0, len(missing_edges) - 1)
            u, v = missing_edges[idx]
            if not (bits[u] & (1 << v)):
                # Try adding
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
                if not find_q3(bits, n):
                    # Accept
                    cur_count = count_edges(bits, n)
                    if cur_count > best_count:
                        best_count = cur_count
                        best_bits = list(bits)
                    # Remove from missing list
                    missing_edges.pop(idx)
                else:
                    # Revert
                    bits[u] &= ~(1 << v)
                    bits[v] &= ~(1 << u)
        
        # Occasionally try to swap: remove one edge, add another
        elif bits and rng.random() < 0.3:
            # Find an edge to remove
            present_edges = [(i, j) for i, j in all_edges if bits[i] & (1 << j)]
            if present_edges and missing_edges:
                u1, v1 = rng.choice(present_edges)
                u2, v2 = rng.choice(missing_edges)
                # Try swap
                bits[u1] &= ~(1 << v1)
                bits[v1] &= ~(1 << u1)
                bits[u2] |= (1 << v2)
                bits[v2] |= (1 << u2)
                if not find_q3(bits, n):
                    cur_count = count_edges(bits, n)
                    if cur_count > best_count:
                        best_count = cur_count
                        best_bits = list(bits)
                    # Update missing list
                    missing_edges.append((u1, v1))
                    # Remove (u2, v2) from missing if present
                    for mi, (a, b) in enumerate(missing_edges):
                        if (a == u2 and b == v2) or (a == v2 and b == u2):
                            missing_edges.pop(mi)
                            break
                else:
                    # Revert
                    bits[u1] |= (1 << v1)
                    bits[v1] |= (1 << u1)
                    bits[u2] &= ~(1 << v2)
                    bits[v2] &= ~(1 << u2)
        
        temp *= cooling
    
    # Phase 3: Greedy improvement - try all missing edges in order
    if best_bits is None:
        best_bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                best_bits[i] |= (1 << j)
                best_bits[j] |= (1 << i)
    
    # Try to add each missing edge greedily
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > deadline - 5.0:
                break
            if not (best_bits[i] & (1 << j)):
                best_bits[i] |= (1 << j)
                best_bits[j] |= (1 << i)
                if not find_q3(best_bits, n):
                    best_count = count_edges(best_bits, n)
                else:
                    best_bits[i] &= ~(1 << j)
                    best_bits[j] &= ~(1 << i)
    
    if best_bits is None:
        return []
    
    return bits_to_edges(best_bits, n)
