import random
import time
import math
from itertools import combinations

def construct():
    deadline = time.time() + 145.0
    n = 40
    target_edges = 401
    
    def compute_books_from_adj(adj):
        """Compute book sizes for all edges using bitmasks."""
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    books[(u,v)] = common.bit_count()
        return books
    
    def max_book_val(books):
        if not books:
            return 0
        return max(books.values())
    
    def build_circulant(offsets):
        """Build a circulant graph with given connection offsets."""
        adj = [0] * n
        for i in range(n):
            for d in offsets:
                j = (i + d) % n
                if j > i:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def count_edges(adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    count += 1
        return count
    
    def evaluate_offset_set(offsets):
        """Evaluate a set of offsets for the circulant graph."""
        adj = build_circulant(offsets)
        books = compute_books_from_adj(adj)
        return max_book_val(books)
    
    # Strategy: Find good circulant offset sets, then add 1 edge and optimize
    
    # Generate candidate offset sets
    # A 10-regular circulant needs 10 offsets (but some may be paired)
    # For n=40, we need offsets that give degree 10
    
    # Let's try various combinations of small offsets
    # The key insight: we want to minimize triangles, so we want
    # offsets that don't create many 3-sums
    
    best_max = float('inf')
    best_adj = None
    best_offset_set = None
    
    # Phase 1: Search for good circulant base
    # Try systematic offset sets
    candidate_sets = []
    
    # Symmetric sets: {±a, ±b, ±c, ±d, ±e}
    for a in range(1, 21):
        for b in range(a+1, 21):
            for c in range(b+1, 21):
                for d in range(c+1, 21):
                    for e in range(d+1, 21):
                        # Check if these give degree 10
                        # Each ±d contributes 2 to degree unless d=0 or d=n/2
                        # For n=40, d=20 gives degree 1 (self-loop issue)
                        # d=0 is invalid
                        offsets = set()
                        for x in [a, b, c, d, e]:
                            offsets.add(x)
                            offsets.add(-x % 40)
                        
                        # Count degree
                        deg = 0
                        for d in offsets:
                            if d != 0 and d != 20:
                                deg += 1
                            elif d == 20:
                                deg += 1  # This creates one edge per vertex
                            # d=0 would be self-loop, skip
                        
                        if deg == 10:
                            candidate_sets.append(frozenset(offsets))
    
    # Also try non-symmetric sets
    # But for simplicity, let's start with symmetric
    
    # Sort by some heuristic: prefer sets with fewer 3-sums
    # A 3-sum x+y+z=0 mod 40 where x,y,z are in the offset set
    
    def count_3sums(offsets):
        """Count number of solutions to x+y+z=0 mod 40 with x,y,z in offsets."""
        count = 0
        offset_list = list(offsets)
        for x in offset_list:
            for y in offset_list:
                for z in offset_list:
                    if (x + y + z) % 40 == 0:
                        count += 1
        return count
    
    # Evaluate top candidates
    # But we can't evaluate all, so let's be smart
    
    # Use a greedy approach: start with a good base and refine
    
    # Known good starting point: offsets that avoid small sums
    # Let's try a few hand-picked sets first
    
    test_sets = [
        frozenset({1, 39, 3, 37, 7, 33, 9, 31, 11, 29}),  # ±1,±3,±7,±9,±11
        frozenset({1, 39, 2, 38, 4, 36, 8, 32, 16, 24}),    # ±1,±2,±4,±8,±16
        frozenset({1, 39, 3, 37, 5, 35, 7, 33, 9, 31}),     # ±1,±3,±5,±7,±9
        frozenset({1, 39, 4, 36, 9, 31, 14, 26, 19, 21}),   # ±1,±4,±9,±14,±19
        frozenset({2, 38, 5, 35, 8, 32, 11, 29, 14, 26}),   # ±2,±5,±8,±11,±14
        frozenset({1, 39, 3, 37, 6, 34, 10, 30, 13, 27}),   # ±1,±3,±6,±10,±13
    ]
    
    for offsets in test_sets:
        if time.time() > deadline - 30:
            break
        max_val = evaluate_offset_set(offsets)
        if max_val < best_max:
            best_max = max_val
            best_offset_set = offsets
    
    # If we found a good base, use it
    if best_offset_set is not None:
        adj = build_circulant(best_offset_set)
        edge_count = count_edges(adj)
        
        # We need 401 edges, so we need to add (401 - edge_count) edges
        # A 10-regular graph on 40 vertices has 40*10/2 = 200 edges
        # So we need to add 201 more edges!
        
        # Wait, that's a lot. Let me reconsider.
        # A k-regular graph on n vertices has n*k/2 edges
        # For n=40, k=10: 40*10/2 = 200 edges
        # We need 401 edges, so we need 201 more edges
        
        # This means the circulant approach gives us only 200 edges
        # We need a much denser graph
        
        # Let's reconsider: maybe we want a higher degree circulant
        # Or use a different approach
        
        # Actually, let's try a different structural approach:
        # Start with K_{20,20} (which has 400 edges) and add 1 edge
        # K_{20,20} has max book size... let's compute
        
        adj_k2020 = [0] * n
        for a in range(20):
            for b in range(20, 40):
                adj_k2020[a] |= (1 << b)
                adj_k2020[b] |= (1 << a)
        
        books_k2020 = compute_books_from_adj(adj_k2020)
        max_k2020 = max_book_val(books_k2020)
        
        # K_{20,20} is bipartite, so it has NO triangles!
        # So max book size is 0
        # But we need 401 edges, so we add 1 edge within one part
        
        # Add edge (0,1)
        adj_k2020[0] |= (1 << 1)
        adj_k2020[1] |= (1 << 0)
        
        books_with_edge = compute_books_from_adj(adj_k2020)
        max_with_edge = max_book_val(books_with_edge)
        
        # The edge (0,1) will be in triangles with every common neighbor
        # 0 and 1 are both in part {0..19}, so their common neighbors
        # are all vertices in {20..39} (20 vertices)
        # So book size for (0,1) is 20
        
        # But wait, are there other edges with large books?
        # Edges within the bipartite graph have book size 0 (no triangles)
        # Only the new edge (0,1) creates triangles
        
        # So max book size is 20
        
        # Can we do better? What if we add the edge differently?
        
        # Actually, any edge within one part will have book size 20
        # because both endpoints connect to all 20 vertices in the other part
        
        # So K_{20,20} + 1 edge always has max book size 20
        
        # The current program achieves score 1.3333, which means max book = 6
        # (since score = seed/value, and seed gives 6, so 1.3333 = 8/6)
        
        # Wait, let me recalculate. The seed program's score is 1.0
        # If seed has max book = 6, then score = 6/6 = 1.0
        # Current program has score 1.3333, so max book = 6/1.3333 = 4.5?
        # That doesn't make sense since book sizes are integers
        
        # Let me re-read: "the search score is seed/value"
        # So if seed value is S and our value is V, score = S/V
        # If score = 1.3333 and seed score = 1.0, then S/S_seed = 1.0
        # So S_seed = S, and S/V = 1.3333, so V = S/1.3333
        
        # If seed has max book = 6, then V = 6/1.3333 = 4.5, impossible
        # So seed must have a different value
        
        # Actually, I think the "seed" in "seed/value" refers to a baseline
        # Let's just focus on minimizing the max book size
        
        # To beat 1.3333, we need max book < 4 (if seed=6) or similar
        
        # Let's try a different approach: start from a random graph
        # and use local search to minimize max book
        
        adj = [0] * n
        edges = set()
        random.seed(42)
        
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u,v) in edges:
                continue
            edges.add((u,v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        books = compute_books_from_adj(adj)
        current_max = max_book_val(books)
        
        if current_max < best_max:
            best_max = current_max
            best_adj = adj[:]
        
        # Local search: try to reduce max book
        improved = True
        no_improve = 0
        
        while improved and time.time() < deadline - 5:
            improved = False
            books = compute_books_from_adj(adj)
            current_max = max_book_val(books)
            
            if current_max < best_max:
                best_max = current_max
                best_adj = adj[:]
            
            # Find edge with max book
            max_edge = None
            max_val = -1
            for e, b in books.items():
                if b > max_val:
                    max_val = b
                    max_edge = e
            
            if max_edge is None:
                break
            
            u, v = max_edge
            common_uv = adj[u] & adj[v]
            common_list = [i for i in range(n) if (common_uv >> i) & 1]
            
            # Try to break triangles by swapping edges
            best_move = None
            best_new_max = current_max
            
            # Try removing (u,w) for w in common_list and adding random edge
            if common_list:
                for w in random.sample(common_list, min(10, len(common_list))):
                    for _ in range(3):
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b:
                            continue
                        if a > b:
                            a, b = b, a
                        if (a,b) == (u,w):
                            continue
                        if (adj[a] >> b) & 1:
                            continue
                        
                        # Apply
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        
                        new_books = compute_books_from_adj(adj)
                        new_max = max_book_val(new_books)
                        
                        # Revert
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                        
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_move = ((u,w), (a,b))
            
            # Also try swapping (u,v) itself
            for _ in range(5):
                a = random.randint(0, n-1)
                b = random.randint(0, n-1)
                if a == b:
                    continue
                if a > b:
                    a, b = b, a
                if (a,b) == (u,v):
                    continue
                if (adj[a] >> b) & 1:
                    continue
                
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                
                new_books = compute_books_from_adj(adj)
                new_max = max_book_val(new_books)
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = ((u,v), (a,b))
            
            if best_move:
                rem, add = best_move
                ru, rv = rem
                au, av = add
                adj[ru] &= ~(1 << rv)
                adj[rv] &= ~(1 << ru)
                adj[au] |= (1 << av)
                adj[av] |= (1 << au)
                improved = True
                no_improve = 0
            else:
                # Random kick
                edges_list = list(edges)
                if edges_list:
                    u2, v2 = random.choice(edges_list)
                    for _ in range(50):
                        a = random.randint(0, n-1)
                        b = random.randint(0, n-1)
                        if a == b:
                            continue
                        if a > b:
                            a, b = b, a
                        if (a,b) == (u2,v2):
                            continue
                        if (adj[a] >> b) & 1:
                            continue
                        
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                        
                        edges.discard((u2,v2))
                        edges.add((a,b))
                        break
                
                no_improve += 1
                if no_improve > 100:
                    break
    
    # Convert best_adj to list of edges
    if best_adj is None:
        # Fallback: return K_{20,20} + 1 edge
        best_adj = [0] * n
        for a in range(20):
            for b in range(20, 40):
                best_adj[a] |= (1 << b)
                best_adj[b] |= (1 << a)
        best_adj[0] |= (1 << 1)
        best_adj[1] |= (1 << 0)
    
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
