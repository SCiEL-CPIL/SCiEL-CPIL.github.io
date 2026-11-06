import time
import random
import numpy as np

def construct():
    deadline = time.time() + 140
    n = 40
    target_edges = 401
    
    def random_graph():
        edges = set()
        while len(edges) < target_edges:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edges.add((min(u,v), max(u,v)))
        return edges
    
    def build_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_books(edges, adj):
        books = {}
        for u, v in edges:
            books[(u,v)] = len(adj[u] & adj[v])
        return books
    
    best_edges = None
    best_score = float('inf')
    
    while time.time() < deadline - 5:
        edges = random_graph()
        adj = build_adj(edges)
        books = compute_books(edges, adj)
        current_score = max(books.values())
        
        # Greedy improvement
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            max_b = max(books.values())
            high_edges = [e for e, b in books.items() if b == max_b]
            remove_edge = random.choice(high_edges)
            u, v = remove_edge
            
            edge_set = set(edges)
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            non_edges = [p for p in all_pairs if p not in edge_set]
            
            best_add = None
            best_new_max = max_b + 1
            sample_size = min(50, len(non_edges))
            candidates = random.sample(non_edges, sample_size)
            
            for add_edge in candidates:
                x, y = add_edge
                new_adj = [s.copy() for s in adj]
                new_adj[u].remove(v)
                new_adj[v].remove(u)
                new_adj[x].add(y)
                new_adj[y].add(x)
                
                new_max = 0
                for e in books:
                    a, b = e
                    if a in (u,v,x,y) or b in (u,v,x,y):
                        nb = len(new_adj[a] & new_adj[b])
                        if nb > new_max:
                            new_max = nb
                ny_book = len(new_adj[x] & new_adj[y])
                if ny_book > new_max:
                    new_max = ny_book
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_add = add_edge
            
            if best_add is not None and best_new_max <= max_b:
                x, y = best_add
                edges = edge_set - {remove_edge} | {best_add}
                adj = [s.copy() for s in adj]
                adj[u].remove(v)
                adj[v].remove(u)
                adj[x].add(y)
                adj[y].add(x)
                books = compute_books(edges, adj)
        
        current_score = max(books.values())
        if current_score < best_score:
            best_score = current_score
            best_edges = edges
            if best_score <= 8:
                break
    
    return [list(e) for e in best_edges]
