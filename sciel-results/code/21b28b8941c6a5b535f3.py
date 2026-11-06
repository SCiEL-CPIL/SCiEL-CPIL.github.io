import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices
    edges = []
    edge_to_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            idx = len(edges)
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            edge_to_idx[(j, i)] = idx
            
    # Precompute K4s
    k4s = []
    k4_edges = [] # list of lists of 6 edge indices
    k4_id = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_to_idx[(a, b)]
                    e2 = edge_to_idx[(a, c)]
                    e3 = edge_to_idx[(a, d)]
                    e4 = edge_to_idx[(b, c)]
                    e5 = edge_to_idx[(b, d)]
                    e6 = edge_to_idx[(c, d)]
                    k4_edges.append([e1, e2, e3, e4, e5, e6])
                    k4_id += 1
                    
    # Precompute which K4s an edge belongs to
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_k4s[e_idx].append(k4_idx)

    def check_k4(k4_idx, colors):
        e_list = k4_edges[k4_idx]
        c_set = set()
        for e in e_list:
            c_set.add(colors[e])
        return len(c_set) >= 5

    def get_violations(colors):
        viol = []
        for k4_idx in range(len(k4_edges)):
            if not check_k4(k4_idx, colors):
                viol.append(k4_idx)
        return viol

    def try_k(K, time_limit):
        # Initialize randomly
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        violations = get_violations(colors)
        
        best_colors = None
        best_viol_count = len(violations)
        
        while time.time() < time_limit and violations:
            # Pick a random violation
            v_idx = random.choice(violations)
            e_list = k4_edges[v_idx]
            
            # Get colors in this K4
            c_vals = [colors[e] for e in e_list]
            c_set = set(c_vals)
            
            # Find a duplicate color
            # Count frequencies
            freq = {}
            for c in c_vals:
                freq[c] = freq.get(c, 0) + 1
            
            dup_color = None
            for c, f in freq.items():
                if f >= 2:
                    dup_color = c
                    break
                    
            if dup_color is None:
                # Should not happen if len(c_set) < 6 and 6 edges
                # If len(c_set) == 5, it's valid.
                # If len(c_set) == 4, max freq is 2 (e.g. 2,2,1,1) or 3, etc.
                # If len(c_set) == 3, max freq is at least 2.
                # So dup_color should exist.
                continue

            # Pick an edge with dup_color
            candidate_edges = [e for e in e_list if colors[e] == dup_color]
            e_to_change = random.choice(candidate_edges)
            
            # Pick a new color not in c_set
            available = [c for c in range(K) if c not in c_set]
            if not available:
                # No available color?
                # This means c_set covers all K colors.
                # But len(c_set) <= 4 (since it's a violation).
                # So K must be <= 4.
                # We are using K >= 13.
                # So this should not happen.
                new_color = random.randint(0, K - 1)
            else:
                new_color = random.choice(available)
                
            old_color = colors[e_to_change]
            colors[e_to_change] = new_color
            
            # Update violations
            # Only check K4s affected by e_to_change
            # Remove v_idx from violations if it becomes valid?
            # Actually, just re-check the affected K4s.
            
            # Mark affected K4s
            affected = edge_k4s[e_to_change]
            
            # We need to update the `violations` list.
            # Simple way: remove all affected from violations, then re-check and add back if invalid.
            # But `violations` is a list.
            
            # Better: maintain a set of violation indices.
            # Let's switch to a set for efficiency if we do this often.
            # For now, let's just re-evaluate the specific v_idx and the others.
            
            # Check v_idx
            if check_k4(v_idx, colors):
                violations.remove(v_idx)
                
            # Check other affected K4s
            # We need to know if they were in violations or not.
            # To avoid complex bookkeeping, let's just re-scan all violations?
            # No, that's slow.
            
            # Let's use a set for violations.
            # Refactor to use set.
            
            pass # Placeholder for logic below

    # Refactored try_k with set
    def try_k_set(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        violations = set(get_violations(colors))
        
        while time.time() < time_limit and violations:
            v_idx = random.choice(list(violations))
            e_list = k4_edges[v_idx]
            
            c_vals = [colors[e] for e in e_list]
            c_set = set(c_vals)
            
            freq = {}
            for c in c_vals:
                freq[c] = freq.get(c, 0) + 1
            
            dup_color = None
            for c, f in freq.items():
                if f >= 2:
                    dup_color = c
                    break
                    
            if dup_color is None:
                continue

            candidate_edges = [e for e in e_list if colors[e] == dup_color]
            e_to_change = random.choice(candidate_edges)
            
            available = [c for c in range(K) if c not in c_set]
            if available:
                new_color = random.choice(available)
            else:
                new_color = random.randint(0, K - 1)
                
            colors[e_to_change] = new_color
            
            # Update violations
            affected = edge_k4s[e_to_change]
            
            # Check v_idx
            if check_k4(v_idx, colors):
                violations.discard(v_idx)
            else:
                violations.add(v_idx) # Already there
            
            # Check other affected
            for k4_idx in affected:
                if k4_idx == v_idx:
                    continue
                # If it was valid, it might become invalid
                # If it was invalid, it might become valid
                if check_k4(k4_idx, colors):
                    violations.discard(k4_idx)
                else:
                    violations.add(k4_idx)
                    
        if not violations:
            # Compress colors
            used_colors = sorted(list(set(colors)))
            mapping = {c: i for i, c in enumerate(used_colors)}
            compressed = [mapping[c] for c in colors]
            return compressed
        return None

    # Try K=13
    if time.time() < deadline - 30:
        res = try_k_set(13, time.time() + 30)
        if res:
            return res

    # Try K=14
    if time.time() < deadline - 20:
        res = try_k_set(14, time.time() + 20)
        if res:
            return res

    # Try K=15
    if time.time() < deadline - 10:
        res = try_k_set(15, time.time() + 10)
        if res:
            return res

    # Fallback: Baseline
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
