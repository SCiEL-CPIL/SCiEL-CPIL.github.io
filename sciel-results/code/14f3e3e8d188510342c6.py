import time
import random

# EVOLVE-BLOCK-START

def _is_valid_full(values):
    """Check if a full sequence has all distinct consecutive block sums."""
    n = len(values)
    seen = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            if total in seen:
                return False
            seen.add(total)
    return True

def _compute_block_sums(values):
    """Compute all consecutive block sums for a sequence."""
    n = len(values)
    sums = set()
    for i in range(n):
        total = 0
        for j in range(i, n):
            total += values[j]
            sums.add(total)
    return sums

def _greedy_forward(values, max_val=256):
    """Greedy: pick smallest valid next element."""
    current = list(values)
    cur_sums = _compute_block_sums(current)
    
    for x in range((current[-1] if current else 0) + 1, max_val + 1):
        tails = [x]
        total = x
        for a in reversed(current):
            total += a
            tails.append(total)
        if not cur_sums.intersection(tails):
            current.append(x)
            cur_sums.update(tails)
    return current

def _build_conflict_graph(max_val=256, max_len=30):
    """
    Build a conflict graph for Maximum Independent Set.
    
    Nodes: (position, value) pairs where position in [0, max_len) and value in [1, max_val]
    Edge between (i, vi) and (j, vj) if they CANNOT both be in a valid sequence.
    
    Two nodes conflict if:
    1. They are at the same position (only one value per position)
    2. The values would create duplicate block sums when both are included
    3. The values violate strictly increasing order (vi < vj for i < j)
    
    Actually, for this problem, we need to think differently.
    The sequence is strictly increasing, so position determines order.
    We pick one value per position, and they must be strictly increasing.
    
    Let's model it as: we have max_len positions. For each position i, we choose a value.
    The values must be strictly increasing: v0 < v1 < v2 < ...
    
    Conflict: Two assignments (i, vi) and (j, vj) with i < j conflict if:
    - vi >= vj (not strictly increasing)
    - OR the block sums created by including both violate distinctness
    
    This is complex. Let's use a different approach.
    """
    pass

def _max_independent_set_tomita(n, adj, deadline=None):
    """
    Tomita's algorithm for Maximum Clique on a graph with n nodes.
    adj is a list of sets, where adj[i] contains all nodes adjacent to i.
    Returns the size of the maximum clique.
    """
    if n == 0:
        return 0
    
    # Convert to 0-indexed adjacency list
    # adj is already 0-indexed
    
    best = [0]
    best_clique = [[]]
    
    def expand(R, P, depth):
        if deadline and time.time() > deadline:
            return
        
        if len(R) > best[0]:
            best[0] = len(R)
            best_clique[0] = list(R)
        
        if not P:
            return
        
        # Greedy coloring for upper bound
        # Order P by degree
        P_list = list(P)
        
        # Simple greedy coloring
        color = {}
        color_class = {}
        for v in P_list:
            # Find the smallest color not used by neighbors in P
            used = set()
            for u in adj[v]:
                if u in P:
                    used.add(color.get(u, -1))
            c = 1
            while c in used:
                c += 1
            color[v] = c
            if c not in color_class:
                color_class[c] = []
            color_class[c].append(v)
        
        # Process color classes in reverse order (highest color first)
        for c in sorted(color_class.keys(), reverse=True):
            if deadline and time.time() > deadline:
                return
            
            # Upper bound: current clique size + c
            if len(R) + c <= best[0]:
                continue
            
            P_c = set(color_class[c])
            for v in P_c:
                if v in P:
                    new_R = R + [v]
                    new_P = P_c & adj[v]
                    expand(new_R, new_P, depth + 1)
                    P_c.discard(v)
                    P.discard(v)
    
    R = []
    P = set(range(n))
    expand(R, P, 0)
    
    return best[0], best_clique[0]

def _solve_with_mis(max_val=256, deadline=None, max_len=30):
    """
    Solve using Maximum Independent Set on a conflict graph.
    
    Nodes: (position, value) where position in [0, max_len) and value in [1, max_val]
    Two nodes (i, vi) and (j, vj) with i < j are compatible if:
    1. vi < vj (strictly increasing)
    2. The block sums created by the partial sequence ending with these values are distinct
    
    We find the maximum independent set, which gives us the longest valid sequence.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    # Build the graph
    # Node ID: position * max_val + value - 1
    # Position i: value must be in [1, max_val]
    
    n_positions = max_len
    n_values = max_val
    total_nodes = n_positions * n_values
    
    def node_id(pos, val):
        return pos * n_values + (val - 1)
    
    # adj[node] = set of compatible nodes
    adj = [set() for _ in range(total_nodes)]
    
    # Precompute compatibility
    # Two nodes (i, vi) and (j, vj) with i < j are compatible if vi < vj
    # AND the block sums don't conflict
    
    # For efficiency, we'll build the adjacency incrementally
    for i in range(n_positions):
        for vi in range(1, max_val + 1):
            node_i = node_id(i, vi)
            for j in range(i + 1, n_positions):
                for vj in range(vi + 1, max_val + 1):
                    node_j = node_id(j, vj)
                    # Check if they are compatible
                    # For now, assume they are compatible if vi < vj
                    # The block sum check is complex, so we'll use a heuristic
                    # Actually, let's just use vi < vj as the primary constraint
                    # and rely on the search to prune invalid sequences
                    adj[node_i].add(node_j)
                    adj[node_j].add(node_i)
    
    # Find maximum clique (which is maximum independent set in the complement)
    # Actually, we want maximum independent set in the conflict graph
    # The conflict graph has edges between incompatible nodes
    # So we want maximum independent set in the conflict graph
    # Which is maximum clique in the complement graph
    
    # Complement graph: edges between compatible nodes
    # So we want maximum clique in the complement graph
    
    # The complement graph has edges where adj has no edge
    # But we built adj as compatible nodes, so adj IS the complement graph
    
    # Find maximum clique in adj
    try:
        clique_size, clique_nodes = _max_independent_set_tomita(total_nodes, adj, deadline)
    except Exception:
        return []
    
    # Convert clique nodes back to sequence
    if not clique_nodes:
        return []
    
    # Sort by position
    clique_sorted = sorted(clique_nodes, key=lambda x: x // n_values)
    
    sequence = []
    for node in clique_sorted:
        pos = node // n_values
        val = (node % n_values) + 1
        sequence.append(val)
    
    return sequence

def _dfs_with_pruning(start_seq, max_val=256, deadline=None, best_len=0):
    """
    DFS with aggressive pruning using sum capacity bounds.
    """
    if deadline and time.time() > deadline:
        return list(start_seq), best_len
    
    best_found = list(start_seq)
    
    def _max_extension_by_sum_capacity(k_curr, t_curr, last_val, max_val=256):
        """Estimate max additional elements."""
        if last_val >= max_val:
            return 0
        
        n_avail = max_val - last_val
        if n_avail == 0:
            return 0
        
        lo, hi = 0, n_avail
        best_m = 0
        
        while lo <= hi:
            mid = (lo + hi) // 2
            k_new = k_curr + mid
            num_sums = k_new * (k_new + 1) // 2
            
            max_add_sum = mid * max_val - mid * (mid - 1) // 2
            t_max = t_curr + max_add_sum
            
            if num_sums <= t_max:
                best_m = mid
                lo = mid + 1
            else:
                hi = mid - 1
                
        return best_m
    
    def dfs(seq, sums_set, total_sum, last, k):
        nonlocal best_found, best_len
        
        if deadline and time.time() > deadline:
            return
        
        if k > best_len:
            best_len = k
            best_found = list(seq)
            
        if last >= max_val:
            return
            
        # Pruning
        max_by_val = max_val - last
        max_by_sum = _max_extension_by_sum_capacity(k, total_sum, last, max_val)
        max_add = min(max_by_val, max_by_sum)
        
        if k + max_add <= best_len:
            return
            
        for x in range(last + 1, max_val + 1):
            if deadline and time.time() > deadline:
                return
            
            tails = [x]
            total = x
            for a in reversed(seq):
                total += a
                tails.append(total)
            
            valid = True
            for t in tails:
                if t in sums_set:
                    valid = False
                    break
            
            if valid:
                new_sums = set(sums_set)
                new_sums.update(tails)
                new_total = total_sum + x
                dfs(seq + [x], new_sums, new_total, x, k + 1)
    
    if start_seq:
        cur_sums = _compute_block_sums(start_seq)
        t_curr = sum(start_seq)
        last_val = start_seq[-1]
        k_curr = len(start_seq)
        dfs(list(start_seq), cur_sums, t_curr, last_val, k_curr)
    else:
        dfs([], set(), 0, 0, 0)
    
    return best_found, best_len

def _beam_search(width=20, max_val=256, deadline=None, max_depth=30):
    """
    Beam search maintaining width most promising partial sequences.
    """
    if deadline is None:
        deadline = time.time() + 50
    
    # Each beam entry: (sequence, sums_set, total_sum, last_val, length)
    beams = [([], set(), 0, 0, 0)]
    
    best_seq = []
    best_len = 0
    
    for depth in range(max_depth):
        if time.time() > deadline:
            break
        
        new_beams = []
        for seq, sums_set, total_sum, last_val, length in beams:
            if last_val >= max_val:
                continue
            
            # Try adding each possible next value
            for x in range(last_val + 1, max_val + 1):
                tails = [x]
                total = x
                for a in reversed(seq):
                    total += a
                    tails.append(total)
                
                if not sums_set.intersection(tails):
                    new_sums = set(sums_set)
                    new_sums.update(tails)
                    new_total = total_sum + x
                    new_seq = seq + [x]
                    new_length = length + 1
                    
                    if new_length > best_len:
                        best_len = new_length
                        best_seq = new_seq
                    
                    new_beams.append((new_seq, new_sums, new_total, x, new_length))
        
        if not new_beams:
            break
        
        # Keep only the width most promising beams
        # Heuristic: prefer longer sequences, then smaller last values (more room)
        new_beams.sort(key=lambda b: (-b[4], b[3]))
        beams = new_beams[:width]
        
        if time.time() > deadline:
            break
    
    return best_seq, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = []
    best_len = 0
    
    # 1. Greedy initial solution
    greedy_seq = _greedy_forward([])
    if len(greedy_seq) > best_len:
        best = greedy_seq
        best_len = len(best)
    
    # 2. Beam search
    if time.time() < deadline - 10:
        beam_seq, beam_len = _beam_search(width=30, max_val=256, deadline=deadline, max_depth=30)
        if beam_len > best_len:
            best = beam_seq
            best_len = beam_len
    
    # 3. DFS with pruning
    if time.time() < deadline - 5:
        found_seq, found_len = _dfs_with_pruning(best, max_val=256, deadline=deadline, best_len=best_len)
        if found_len > best_len:
            best = found_seq
            best_len = found_len
    
    # 4. Randomized restarts
    if time.time() < deadline - 3:
        rng = random.Random(12345)
        for _ in range(5):
            if time.time() > deadline:
                break
            
            # Random prefix
            prefix = []
            cur_sums = set()
            last = 0
            for _ in range(2):
                found = False
                for _ in range(100):
                    x = rng.randint(last + 1, 256)
                    tails = [x]
                    total = x
                    for a in reversed(prefix):
                        total += a
                        tails.append(total)
                    if not cur_sums.intersection(tails):
                        prefix.append(x)
                        cur_sums.update(tails)
                        last = x
                        found = True
                        break
                if not found:
                    break
            
            if len(prefix) >= 2:
                found_seq, found_len = _dfs_with_pruning(prefix, max_val=256, deadline=deadline, best_len=best_len)
                if found_len > best_len:
                    best = found_seq
                    best_len = found_len
    
    # Final validation
    if not _is_valid_full(best):
        best = _greedy_forward([])
    
    return best

# EVOLVE-BLOCK-END
