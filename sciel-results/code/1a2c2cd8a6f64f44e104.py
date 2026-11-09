import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    t_start = time.time()
    deadline = t_start + 110

    def compute_books(adj_masks):
        """Compute max book size from bitmask neighbor lists"""
        max_book = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj_masks[i] & (1 << j):
                    common = adj_masks[i] & adj_masks[j]
                    c = common.bit_count()
                    if c > max_book:
                        max_book = c
        return max_book

    def structural_init():
        """
        Structural initialization inspired by the high-performing solution:
        Split into A (small) and B (large), complete bipartite A-B,
        then add edges within B with degree constraints to minimize common neighbors.
        """
        # Split vertices into two parts: A (small) and B (large)
        # Try a few different splits to find the best structure
        best_bits = None
        best_score = 100
        
        for a_size in [8, 9, 10, 11]:
            A = list(range(a_size))
            B = list(range(a_size, n))
            
            adj = [0] * n
            edge_set = set()
            
            # Add all edges between A and B
            for a in A:
                for b in B:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
                    edge_set.add((a, b))
            
            # Need m - a_size * (n - a_size) more edges within B
            needed = m - a_size * (n - a_size)
            
            if needed < 0:
                continue
            
            # Split B into B1 and B2 for balanced distribution
            b_size = n - a_size
            b1_size = b_size // 2
            B1 = B[:b1_size]
            B2 = B[b1_size:]
            
            # Generate all possible pairs within B
            pairs = []
            for u in B:
                for v in B:
                    if u < v:
                        pairs.append((u, v))
            random.shuffle(pairs)
            
            deg = {v: 0 for v in B}
            added = 0
            max_deg = 10  # Degree constraint to keep structure sparse
            
            for u, v in pairs:
                if added == needed:
                    break
                if deg[u] < max_deg and deg[v] < max_deg:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    deg[u] += 1
                    deg[v] += 1
                    added += 1
            
            if added < needed:
                # Fill remaining ignoring constraints
                for u, v in pairs:
                    if added == needed:
                        break
                    if not (adj[u] & (1 << v)):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        edge_set.add((u, v))
                        added += 1
            
            if added != needed:
                continue
            
            score = compute_books(adj)
            if score < best_score:
                best_score = score
                best_bits = adj[:]
        
        return best_bits, best_score

    def random_init():
        """Random graph with m edges"""
        adj = [0] * n
        edge_set = set()
        while len(edge_set) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) not in edge_set:
                edge_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj, edge_set, compute_books(adj)

    def swap_edge(adj, edge_set, out_u, out_v, in_u, in_v):
        """Swap out edge (out_u,out_v) for (in_u,in_v)"""
        adj[out_u] &= ~(1 << out_v)
        adj[out_v] &= ~(1 << out_u)
        edge_set.discard((out_u, out_v))
        edge_set.discard((out_v, out_u))

        adj[in_u] |= (1 << in_v)
        adj[in_v] |= (1 << in_u)
        edge_set.add((min(in_u, in_v), max(in_u, in_v)))

    def get_max_book_info(adj, edge_set):
        """Get max book size and the edges achieving it"""
        max_b = -1
        max_edges = []
        for u, v in edge_set:
            b = (adj[u] & adj[v]).bit_count()
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    best_score = 100
    best_adj = None

    # Phase 1: Structural initialization
    struct_adj, struct_score = structural_init()
    if struct_adj is not None:
        best_score = min(best_score, struct_score)
        best_adj = struct_adj[:]
    
    if best_score <= 7:
        # Convert to edge list
        final_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i] & (1 << j):
                    final_edges.append([i, j])
        return final_edges

    # Phase 2: Local search with smart edge selection
    # Start from the best structural init
    current_adj = best_adj[:]
    current_score = best_score
    
    # Rebuild edge_set
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if current_adj[i] & (1 << j):
                edge_set.add((i, j))

    last_improve = time.time()
    no_improve_count = 0
    max_no_improve = 500

    while time.time() < deadline:
        # Stagnation check
        if time.time() - last_improve > 10.0:
            # Try different initialization
            strategy = random.random()
            if strategy < 0.5:
                struct_adj, struct_score = structural_init()
                if struct_adj is not None:
                    current_adj = struct_adj[:]
                    current_score = struct_score
            else:
                current_adj, edge_set, current_score = random_init()
            
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
            
            last_improve = time.time()
            no_improve_count = 0
            continue

        # Get current max book info
        current_max, current_max_edges = get_max_book_info(current_adj, edge_set)
        
        if current_max <= 7:
            if current_max == 7 and no_improve_count > 200:
                break
            elif current_max <= 6:
                break

        if not current_max_edges:
            break

        # Pick an edge from the max-book set to remove
        out_u, out_v = random.choice(current_max_edges)
        
        # Find the best non-edge to add: minimize book size of new edge
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        
        # Iterate over all possible non-edges
        for x in range(n):
            for y in range(x + 1, n):
                # Check if (x,y) is already an edge
                if (1 << y) & current_adj[x]:
                    continue
                
                # Calculate book size for potential new edge
                b = (current_adj[x] & current_adj[y]).bit_count()
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
                    if b == 0:
                        break
            if min_book_new == 0:
                break
        
        if best_x == -1:
            # No non-edges? Very unlikely with 401 edges out of 780
            break
        
        # Perform the swap: remove (out_u, out_v), add (best_x, best_y)
        old_max = current_max
        old_max_count = len(current_max_edges)
        
        # Apply swap
        swap_edge(current_adj, edge_set, out_u, out_v, best_x, best_y)
        
        # Recalculate max book
        new_max, new_max_edges = get_max_book_info(current_adj, edge_set)
        
        # Acceptance criteria
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            # Prefer moves that reduce the number of max-book edges
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                # Small probability to accept to escape local minima
                if random.random() < 0.05:
                    accepted = True
                else:
                    accepted = False
        else:
            accepted = False

        if accepted:
            current_score = new_max
            if current_score < best_score:
                best_score = current_score
                best_adj = current_adj[:]
                last_improve = time.time()
                if best_score <= 7:
                    break
        else:
            # Rollback
            swap_edge(current_adj, edge_set, best_x, best_y, out_u, out_v)
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturb: do a few random swaps
                for _ in range(5):
                    if edge_set:
                        ru, rv = random.choice(list(edge_set))
                        while True:
                            xu = random.randint(0, n - 1)
                            xv = random.randint(0, n - 1)
                            if xu == xv:
                                continue
                            if xu > xv:
                                xu, xv = xv, xu
                            if (xu, xv) == (ru, rv):
                                continue
                            if not (current_adj[xu] & (1 << xv)):
                                break
                        swap_edge(current_adj, edge_set, ru, rv, xu, xv)
                no_improve_count = 0
                last_improve = time.time()

    # Reconstruct edge list from best_adj
    if best_adj is not None:
        final_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i] & (1 << j):
                    final_edges.append([i, j])
        return final_edges
    else:
        # Fallback: return current edge set
        return [[u, v] for u, v in sorted(edge_set)]
