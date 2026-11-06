import time
import random

def _index(i, j, n):
    """Lexicographic index of pair (i,j) with i<j in the flat list."""
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for each color. adj[c][v] is a bitset of neighbors of v in color c."""
    adj = [[0] * n for _ in range(5)]
    for i in range(n):
        for j in range(i + 1, n):
            idx = _index(i, j, n)
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    return adj

def _count_c4s(adj, n):
    """Count total number of C4 violations across all colors."""
    count = 0
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):
                    count += 1
    return count

def _find_violation(adj, n):
    """Find one C4 violation. Returns (c, u, v, n1, n2) or None."""
    for c in range(5):
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[c][u] & adj[c][v]
                if common & (common - 1):
                    bits = common
                    b1 = bits & (-bits)
                    bits ^= b1
                    b2 = bits & (-bits)
                    n1 = b1.bit_length() - 1
                    n2 = b2.bit_length() - 1
                    return (c, u, v, n1, n2)
    return None

def _is_valid(colors, n):
    """Check if coloring is valid (no monochromatic C4)."""
    adj = _build_adj(colors, n)
    return _count_c4s(adj, n) == 0

def construct():
    deadline = time.time() + 145  # Safety margin
    
    # Fallback: Known 11-vertex construction
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to find valid colorings for n=26 down to 12
    # Focus heavily on 26 as it's the known algebraic bound
    for target_n in [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if time.time() > deadline - 5:
            break
        
        num_edges = target_n * (target_n - 1) // 2
        
        # Number of attempts: more for larger n? Or just fixed high number?
        # For n=26, try many restarts.
        max_attempts = 500 if target_n >= 24 else 100
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                break
            
            # Random initialization
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = _build_adj(colors, target_n)
            current_violations = _count_c4s(adj, target_n)
            
            # Local Search
            max_iters = 5000
            stuck_count = 0
            for it in range(max_iters):
                if time.time() > deadline - 1:
                    break
                
                if current_violations == 0:
                    # Valid!
                    best_n = target_n
                    best_colors = colors[:]
                    break
                
                # Find a violation
                viol = _find_violation(adj, target_n)
                if viol is None:
                    best_n = target_n
                    best_colors = colors[:]
                    break
                
                c, u, v, n1, n2 = viol
                
                # Try to fix by recoloring one of the edges in the C4
                # Edges: (u,n1), (v,n1), (u,n2), (v,n2)
                # Also potentially (u,v) if it was part of the structure? No, C4 is u-n1-v-n2-u.
                # The edges of the C4 are (u,n1), (n1,v), (v,n2), (n2,u).
                # In our adjacency, (u,v) is NOT an edge of the C4 unless it's a triangle, but C4 is 4-cycle.
                # So we recolor one of the 4 edges of the cycle.
                
                candidate_edges = [
                    (min(u, n1), max(u, n1)),
                    (min(v, n1), max(v, n1)),
                    (min(u, n2), max(u, n2)),
                    (min(v, n2), max(v, n2))
                ]
                
                random.shuffle(candidate_edges)
                
                improved = False
                best_delta = 1
                best_move = None
                
                for (a, b) in candidate_edges:
                    idx = _index(a, b, target_n)
                    old_c = colors[idx]
                    
                    for new_c in range(5):
                        if new_c == old_c:
                            continue
                        
                        # Apply change
                        adj[old_c][a] &= ~(1 << b)
                        adj[old_c][b] &= ~(1 << a)
                        adj[new_c][a] |= (1 << b)
                        adj[new_c][b] |= (1 << a)
                        
                        new_count = _count_c4s(adj, target_n)
                        delta = new_count - current_violations
                        
                        # Revert
                        adj[new_c][a] &= ~(1 << b)
                        adj[new_c][b] &= ~(1 << a)
                        adj[old_c][a] |= (1 << b)
                        adj[old_c][b] |= (1 << a)
                        
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (idx, old_c, new_c, a, b)
                
                if best_move is not None and best_delta < 0:
                    idx, old_c, new_c, a, b = best_move
                    adj[old_c][a] &= ~(1 << b)
                    adj[old_c][b] &= ~(1 << a)
                    adj[new_c][a] |= (1 << b)
                    adj[new_c][b] |= (1 << a)
                    colors[idx] = new_c
                    current_violations += best_delta
                    stuck_count = 0
                else:
                    # No improving move found. Random move to escape.
                    stuck_count += 1
                    if stuck_count > 50:
                        # Restart this attempt
                        break
                    
                    idx = random.randint(0, num_edges - 1)
                    # Find edge for idx
                    # Precompute edge list for speed? No, n is small.
                    edge_found = False
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            if _index(i, j, target_n) == idx:
                                old_c = colors[idx]
                                new_c = random.randint(0, 4)
                                if new_c != old_c:
                                    adj[old_c][i] &= ~(1 << j)
                                    adj[old_c][j] &= ~(1 << i)
                                    adj[new_c][i] |= (1 << j)
                                    adj[new_c][j] |= (1 << i)
                                    colors[idx] = new_c
                                    current_violations = _count_c4s(adj, target_n)
                                edge_found = True
                                break
                        if edge_found:
                            break
            else:
                continue
            
            if best_n == target_n:
                break
        
        if best_n == target_n:
            break

    return {'n': best_n, 'colors': best_colors}
