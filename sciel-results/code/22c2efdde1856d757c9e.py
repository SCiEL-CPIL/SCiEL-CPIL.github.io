"""Rewritten program: random init + local search to minimize largest book."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    n = 40
    m = 401
    deadline = time.time() + 140  # safety margin before 150s kill

    def compute_books(adj, edge_list):
        """Return dict edge -> book size (common neighbors)."""
        books = {}
        for u, v in edge_list:
            common = adj[u] & adj[v]
            books[(u, v)] = len(common)
        return books

    def max_book(books):
        return max(books.values()) if books else 0

    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def smart_init():
        """Start with a near-balanced bipartite graph plus random extras.
        
        Bipartite K_20,20 has 400 edges, 0 triangles. Add 1 more edge
        within one part to get 401 edges with minimal triangles.
        """
        edges = set()
        # Create bipartition: A = 0..19, B = 20..39
        for u in range(20):
            for v in range(20, 40):
                edges.add((u, v))
        # Need 1 more edge; add an edge within A that minimizes new triangles
        # Edge (0,1) in A: triangles formed = common neighbors of 0 and 1 in B = all 20
        # That's bad. Instead, let's add an edge and then use local search to fix.
        # Actually, any intra-part edge creates 20 triangles immediately.
        # Better: start with 400 bipartite edges, add one random edge, then local search.
        # Or: start slightly differently - remove a few bipartite edges and add intra-part edges
        # that create fewer triangles.
        
        # Alternative: start with a random graph but biased toward low-degree structure
        # Let's just use random for now but with a better distribution
        edges = set()
        while len(edges) < m:
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return list(edges)

    def random_graph():
        edges = set()
        while len(edges) < m:
            u = random.randrange(n)
            v = random.randrange(n)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return list(edges)

    # Start with a random graph
    edge_list = random_graph()
    adj = build_adj(edge_list)
    books = compute_books(adj, edge_list)
    best_max = max_book(books)

    # Local search: try edge swaps to reduce max book
    edge_set = set(edge_list)
    
    # Local search loop with incremental updates
    improvement = True
    stuck_count = 0
    while improvement and time.time() < deadline:
        improvement = False
        stuck_count = 0
        
        # Get edges sorted by book size descending
        sorted_edges = sorted(books.items(), key=lambda x: -x[1])
        
        # Try to improve high-book edges
        for (u, v), bk in sorted_edges:
            if time.time() >= deadline:
                break
            if bk < best_max:
                break  # no more edges with book >= best_max to improve
            
            # Try removing (u,v) and adding a non-edge that creates low book
            # Collect candidate non-edges and evaluate them
            best_candidate = None
            best_candidate_book = bk  # must be strictly less than bk to accept
            
            # Try many random non-edges
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts and time.time() < deadline:
                attempts += 1
                c = random.randrange(n)
                d = random.randrange(n)
                if c == d:
                    continue
                c, d = min(c, d), max(c, d)
                if (c, d) in edge_set:
                    continue
                if (c, d) == (u, v):
                    continue
                
                # Compute common neighbors for (c,d) after removing (u,v)
                ac = adj[c]
                ad = adj[d]
                if c == u:
                    ac = adj[u] - {v}
                elif c == v:
                    ac = adj[v] - {u}
                else:
                    ac = adj[c]
                
                if d == u:
                    ad = adj[u] - {v}
                elif d == v:
                    ad = adj[v] - {u}
                else:
                    ad = adj[d]
                
                common_cd = len(ac & ad)
                
                if common_cd < best_candidate_book:
                    best_candidate_book = common_cd
                    best_candidate = (c, d)
            
            if best_candidate is not None:
                c, d = best_candidate
                
                # Perform the swap
                edge_set.discard((u, v))
                edge_set.add((c, d))
                adj[u].discard(v)
                adj[v].discard(u)
                adj[c].add(d)
                adj[d].add(c)
                
                # Recompute books for affected edges only
                # Edges affected: all edges incident to u, v, c, d
                # For simplicity, recompute all books (n=40, m=401, this is fast enough)
                edge_list = list(edge_set)
                books = compute_books(adj, edge_list)
                new_max = max_book(books)
                
                if new_max <= best_max:
                    best_max = new_max
                    improvement = True
                else:
                    # Revert
                    edge_set.discard((c, d))
                    edge_set.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
                    adj[c].discard(d)
                    adj[d].discard(c)
                    edge_list = list(edge_set)
                    books = compute_books(adj, edge_list)
                    stuck_count += 1
                
                # If we made an improvement, restart the outer loop
                if improvement:
                    break
            
            # If we couldn't improve this edge, try next
            # But also break if we've been stuck too long
            if stuck_count > 50 and time.time() < deadline:
                # Perturb: do a random swap to escape local minimum
                if len(edge_set) >= 2:
                    el = list(edge_set)
                    a, b = random.choice(el)
                    # Find a non-edge to swap in
                    for _ in range(100):
                        c = random.randrange(n)
                        d = random.randrange(n)
                        if c == d:
                            continue
                        c, d = min(c, d), max(c, d)
                        if (c, d) not in edge_set and (c, d) != (a, b):
                            edge_set.discard((a, b))
                            edge_set.add((c, d))
                            adj[a].discard(b)
                            adj[b].discard(a)
                            adj[c].add(d)
                            adj[d].add(c)
                            edge_list = list(edge_set)
                            books = compute_books(adj, edge_list)
                            new_max = max_book(books)
                            if new_max <= best_max + 1:  # allow slight worsening
                                best_max = new_max
                                improvement = True
                            else:
                                # Revert
                                edge_set.discard((c, d))
                                edge_set.add((a, b))
                                adj[a].add(b)
                                adj[b].add(a)
                                adj[c].discard(d)
                                adj[d].discard(c)
                                edge_list = list(edge_set)
                                books = compute_books(adj, edge_list)
                            break
    
    # Final polish: try more exhaustive local moves
    if time.time() < deadline - 10:
        for _ in range(5):
            if time.time() >= deadline:
                break
            improved = False
            sorted_edges = sorted(books.items(), key=lambda x: -x[1])
            for (u, v), bk in sorted_edges:
                if time.time() >= deadline:
                    break
                if bk < best_max:
                    break
                
                # Try all non-edges (more exhaustive)
                best_cand = None
                best_cand_book = bk
                for c in range(n):
                    for d in range(c + 1, n):
                        if (c, d) in edge_set or (c, d) == (u, v):
                            continue
                        
                        ac = adj[c]
                        ad = adj[d]
                        if c == u:
                            ac = adj[u] - {v}
                        elif c == v:
                            ac = adj[v] - {u}
                        if d == u:
                            ad = adj[u] - {v}
                        elif d == v:
                            ad = adj[v] - {u}
                        
                        common_cd = len(ac & ad)
                        if common_cd < best_cand_book:
                            best_cand_book = common_cd
                            best_cand = (c, d)
                
                if best_cand is not None:
                    c, d = best_cand
                    edge_set.discard((u, v))
                    edge_set.add((c, d))
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[c].add(d)
                    adj[d].add(c)
                    
                    edge_list = list(edge_set)
                    books = compute_books(adj, edge_list)
                    new_max = max_book(books)
                    
                    if new_max <= best_max:
                        best_max = new_max
                        improved = True
                    else:
                        edge_set.discard((c, d))
                        edge_set.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        adj[c].discard(d)
                        adj[d].discard(c)
                        edge_list = list(edge_set)
                        books = compute_books(adj, edge_list)
            
            if not improved:
                break

    # Final: return the best edge list found
    edge_list = list(edge_set)
    return [[u, v] for u, v in edge_list]
# EVOLVE-BLOCK-END
