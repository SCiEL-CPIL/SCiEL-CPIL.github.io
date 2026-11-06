"""Construct a large (5,5)-free graph via optimized randomized local search."""

import time
import random
import math

def _popcount(x):
    """Count bits in integer."""
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def _find_k5_and_score(n, adj_masks):
    """Find a K5. Returns (k5_tuple, score) where score measures how 'tight' the K5 is."""
    for v in range(n):
        nbr_mask = adj_masks[v]
        nodes = []
        m = nbr_mask
        while m:
            lsb = m & -m
            nodes.append(lsb.bit_length() - 1)
            m ^= lsb
        k = len(nodes)
        if k < 4:
            continue
        for i in range(k):
            a = nodes[i]
            for j in range(i + 1, k):
                b = nodes[j]
                common = nbr_mask & adj_masks[a] & adj_masks[b]
                c_mask = common
                while c_mask:
                    lsb = c_mask & -c_mask
                    c = lsb.bit_length() - 1
                    c_mask ^= lsb
                    d_mask = common & adj_masks[c]
                    if d_mask:
                        lsb_d = d_mask & -d_mask
                        d = lsb_d.bit_length() - 1
                        return (v, a, b, c, d)
    return None, 0

def _find_empty5(n, adj_masks):
    """Find an independent set of size 5."""
    all_mask = (1 << n) - 1
    for v in range(n):
        non_nbr_mask = all_mask & ~(adj_masks[v] | (1 << v))
        nodes = []
        m = non_nbr_mask
        while m:
            lsb = m & -m
            nodes.append(lsb.bit_length() - 1)
            m ^= lsb
        k = len(nodes)
        if k < 4:
            continue
        for i in range(k):
            a = nodes[i]
            for j in range(i + 1, k):
                b = nodes[j]
                common_non = non_nbr_mask & ~adj_masks[a] & ~adj_masks[b]
                c_mask = common_non
                while c_mask:
                    lsb = c_mask & -c_mask
                    c = lsb.bit_length() - 1
                    c_mask ^= lsb
                    d_mask = common_non & ~adj_masks[c] & ~(1 << c)
                    if d_mask:
                        lsb_d = d_mask & -d_mask
                        d = lsb_d.bit_length() - 1
                        return (v, a, b, c, d)
    return None

def _count_violations_fast(n, adj_masks, cap=100):
    """Fast count of K5s and Empty5s, capped."""
    count = 0
    # Count K5s
    for v in range(n):
        nbr_mask = adj_masks[v]
        nodes = []
        m = nbr_mask
        while m:
            lsb = m & -m
            nodes.append(lsb.bit_length() - 1)
            m ^= lsb
        k = len(nodes)
        if k >= 4:
            for i in range(k):
                a = nodes[i]
                for j in range(i + 1, k):
                    b = nodes[j]
                    common = nbr_mask & adj_masks[a] & adj_masks[b]
                    c_mask = common
                    while c_mask:
                        lsb = c_mask & -c_mask
                        c = lsb.bit_length() - 1
                        c_mask ^= lsb
                        if common & adj_masks[c]:
                            count += 1
                            if count >= cap:
                                return count
    # Count Empty5s
    all_mask = (1 << n) - 1
    for v in range(n):
        non_nbr_mask = all_mask & ~(adj_masks[v] | (1 << v))
        nodes = []
        m = non_nbr_mask
        while m:
            lsb = m & -m
            nodes.append(lsb.bit_length() - 1)
            m ^= lsb
        k = len(nodes)
        if k >= 4:
            for i in range(k):
                a = nodes[i]
                for j in range(i + 1, k):
                    b = nodes[j]
                    common_non = non_nbr_mask & ~adj_masks[a] & ~adj_masks[b]
                    c_mask = common_non
                    while c_mask:
                        lsb = c_mask & -c_mask
                        c = lsb.bit_length() - 1
                        c_mask ^= lsb
                        d_mask = common_non & ~adj_masks[c] & ~(1 << c)
                        if d_mask:
                            count += 1
                            if count >= cap:
                                return count
    return count

def _edges_from_masks(n, adj_masks):
    """Convert adjacency masks to edge list."""
    edges = []
    for i in range(n):
        m = adj_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append((i, j))
            m ^= lsb
    return edges

def construct():
    deadline = time.time() + 295  # Safety margin before 330s kill
    
    best_n = 37
    best_edges = None
    
    # Try to find valid graph for n from 44 down to 38
    for n in range(44, 37, -1):
        if time.time() > deadline:
            break
        
        # Try multiple strategies for this n
        found = False
        
        # Strategy 1: Random restarts with simulated annealing
        densities = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for density in densities:
            if time.time() > deadline or found:
                break
            
            for attempt in range(8):
                if time.time() > deadline or found:
                    break
                
                # Initialize random graph
                adj_masks = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < density:
                            adj_masks[i] |= (1 << j)
                            adj_masks[j] |= (1 << i)
                
                current_violations = _count_violations_fast(n, adj_masks)
                
                if current_violations == 0:
                    best_n = n
                    best_edges = _edges_from_masks(n, adj_masks)
                    found = True
                    break
                
                # Simulated annealing
                temp = 3.0
                cooling = 0.9995
                min_temp = 0.05
                steps = 3000
                
                for step in range(steps):
                    if time.time() > deadline:
                        break
                    
                    if temp < min_temp:
                        break
                    
                    # Try to find and fix a violation
                    k5, _ = _find_k5_and_score(n, adj_masks)
                    if k5:
                        verts = list(k5)
                        random.shuffle(verts)
                        i, j = verts[0], verts[1]
                        
                        # Try removing edge (i,j)
                        adj_masks[i] ^= (1 << j)
                        adj_masks[j] ^= (1 << i)
                        
                        new_violations = _count_violations_fast(n, adj_masks)
                        delta = new_violations - current_violations
                        
                        if delta <= 0 or (delta > 0 and random.random() < math.exp(-delta / temp)):
                            current_violations = new_violations
                            if current_violations == 0:
                                best_n = n
                                best_edges = _edges_from_masks(n, adj_masks)
                                found = True
                                break
                        else:
                            adj_masks[i] ^= (1 << j)
                            adj_masks[j] ^= (1 << i)
                        
                        temp *= cooling
                        continue

                    e5 = _find_empty5(n, adj_masks)
                    if e5:
                        verts = list(e5)
                        random.shuffle(verts)
                        i, j = verts[0], verts[1]
                        
                        # Try adding edge (i,j)
                        adj_masks[i] ^= (1 << j)
                        adj_masks[j] ^= (1 << i)
                        
                        new_violations = _count_violations_fast(n, adj_masks)
                        delta = new_violations - current_violations
                        
                        if delta <= 0 or (delta > 0 and random.random() < math.exp(-delta / temp)):
                            current_violations = new_violations
                            if current_violations == 0:
                                best_n = n
                                best_edges = _edges_from_masks(n, adj_masks)
                                found = True
                                break
                        else:
                            adj_masks[i] ^= (1 << j)
                            adj_masks[j] ^= (1 << i)
                        
                        temp *= cooling
                        continue
                        
                    # No violation found by finder but count > 0 - unlikely, break
                    break
        
        if found:
            break
    
    # Fallback: Paley graph on 37 vertices
    if best_edges is None:
        n = 37
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        best_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    return best_n, best_edges
