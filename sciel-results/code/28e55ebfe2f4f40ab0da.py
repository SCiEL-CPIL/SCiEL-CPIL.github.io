import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds buffer
    
    # 1. Generate STS(15)
    # Points 0..14 correspond to 1..15 in GF(2)^4
    sts_triples = []
    for i in range(1, 16):
        for j in range(i+1, 16):
            k = i ^ j
            if k > j:
                t = tuple(sorted([i-1, j-1, k-1]))
                sts_triples.append(t)
    
    # Convert to sets for faster intersection checks? 
    # Or keep as tuples and use set membership.
    
    # 2. Find all 8-caps
    all_caps = []
    for combo in itertools.combinations(range(15), 8):
        c_set = set(combo)
        is_cap = True
        for t in sts_triples:
            if t[0] in c_set and t[1] in c_set and t[2] in c_set:
                is_cap = False
                break
        if is_cap:
            all_caps.append(combo)
            
    if not all_caps:
        # Should not happen
        return []

    # 3. Precompute cap hits for each pair
    # pair -> list of cap indices
    pair_to_caps = {}
    for u in range(15):
        for v in range(u+1, 15):
            pair_to_caps[(u,v)] = []
    
    for idx, cap in enumerate(all_caps):
        cap_set = set(cap)
        for u in cap:
            for v in cap:
                if u < v:
                    pair_to_caps[(u,v)].append(idx)
                    
    # 4. Find a matching M hitting all caps
    # Use backtracking with heuristic
    
    n_caps = len(all_caps)
    hit_mask = 0  # Bitmask of hit caps? 6435 bits is too big for int? 
    # Python ints are arbitrary precision, so bitmask is fine.
    
    # Initialize bitmask for each pair
    pair_hit_mask = {}
    for pair, cap_indices in pair_to_caps.items():
        mask = 0
        for idx in cap_indices:
            mask |= (1 << idx)
        pair_hit_mask[pair] = mask
        
    all_caps_mask = (1 << n_caps) - 1
    
    best_matching = None
    
    # Sort pairs by number of caps hit (descending) for heuristic
    # But we need to respect matching constraint.
    
    def search(available, current_matching, current_hit_mask, depth):
        nonlocal best_matching
        if time.time() > deadline:
            return
        
        if current_hit_mask == all_caps_mask:
            best_matching = current_matching[:]
            return
            
        # If no caps left to hit, we are done? No, we must hit ALL.
        # If current_hit_mask covers all, we return.
        
        # Find an unhit cap
        # We can pick the cap with the fewest available pairs (MRV heuristic)
        # Or just pick the first unhit cap.
        
        # Let's pick the first unhit cap for simplicity, or use MRV.
        # MRV: Find cap index i such that (1<<i) not in current_hit_mask.
        # Among pairs in that cap that are available, pick the one with max coverage?
        
        # For speed, let's just iterate through caps.
        # To make it faster, we can maintain a list of unhit caps.
        
        # Simple heuristic: Pick the first unhit cap.
        unhit_cap_idx = None
        for i in range(n_caps):
            if not (current_hit_mask & (1 << i)):
                unhit_cap_idx = i
                break
                
        if unhit_cap_idx is None:
            return # Should not happen if check above fails
            
        cap = all_caps[unhit_cap_idx]
        
        # Try all pairs in this cap
        # Sort pairs by how many NEW caps they hit?
        pairs_in_cap = []
        for u in cap:
            for v in cap:
                if u < v:
                    if u in available and v in available:
                        pairs_in_cap.append((u,v))
                        
        # Heuristic: sort by number of caps hit by this pair (global)
        # Or by number of unhit caps hit.
        # Let's sort by global count first.
        pairs_in_cap.sort(key=lambda p: len(pair_to_caps[p]), reverse=True)
        
        for p in pairs_in_cap:
            u, v = p
            # Check if u, v are available
            if u not in available or v not in available:
                continue
                
            new_matching = current_matching + [p]
            new_available = available - {u, v}
            new_hit_mask = current_hit_mask | pair_hit_mask[p]
            
            search(new_available, new_matching, new_hit_mask, depth + 1)
            
            if best_matching is not None:
                return

    # Start search
    available = set(range(15))
    search(available, [], 0, 0)
    
    if best_matching is None:
        # Fallback: Use the previous matching strategy or a random one
        # If we didn't find a perfect matching, we might still find a good one.
        # For now, return a system that might have alpha=9 or 10.
        # Let's try to find a matching that hits MOST caps.
        # Or just use the fallback from the original code.
        found_M = [(i, i+1) for i in range(0, 14, 2)]
    else:
        found_M = best_matching

    # 5. Construct final triples
    M_pairs = [set(m) for m in found_M]
    
    final_triples = []
    
    # Keep STS triples that do not contain any pair in M
    for t in sts_triples:
        t_set = set(t)
        keep = True
        for m_set in M_pairs:
            if m_set.issubset(t_set):
                keep = False
                break
        if keep:
            final_triples.append(list(t))
            
    # Add new triples with 15
    for m in found_M:
        final_triples.append([15, m[0], m[1]])
        
    return final_triples
