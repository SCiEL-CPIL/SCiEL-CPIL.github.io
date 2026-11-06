import time
import random

def construct():
    deadline = time.time() + 140
    n = 40
    target_edges = 401
    
    def compute_books(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        books = {}
        for u, v in edges:
            books[(u, v)] = len(adj[u] & adj[v])
        return books
    
    def max_book(books):
        return max(books.values()) if books else 0
    
    random.seed(42)
    
    def init_structured():
        edges = set()
        for a in range(20):
            for b in range(20, 40):
                edges.add((a, b))
        bip_edges = list(edges)
        random.shuffle(bip_edges)
        for e in bip_edges[:15]:
            edges.discard(e)
        part_a_edges = [(i, j) for i in range(20) for j in range(i+1, 20)]
        part_b_edges = [(20+i, 20+j) for i in range(20) for j in range(i+1, 20)]
        all_intra = part_a_edges + part_b_edges
        random.shuffle(all_intra)
        for e in all_intra[:15]:
            edges.add(e)
        return edges
    
    def init_random():
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        return set(all_possible[:target_edges])
    
    best_edges = None
    best_score = float('inf')
    
    for init_type in ['structured', 'random', 'structured', 'random']:
        if time.time() >= deadline:
            break
        if init_type == 'structured':
            edges = init_structured()
        else:
            edges = init_random()
        
        books = compute_books(edges)
        score = max_book(books)
        
        if score < best_score:
            best_score = score
            best_edges = set(edges)
        
        while time.time() < deadline:
            max_b = max_book(books)
            
            if max_b <= 7:
                break
            
            worst = [e for e, b in books.items() if b == max_b]
            random.shuffle(worst)
            
            improved = False
            for (u, v) in worst[:3]:
                if time.time() >= deadline:
                    break
                new_edges = edges - {(u, v)}
                adj = [set() for _ in range(n)]
                for a, b in new_edges:
                    adj[a].add(b)
                    adj[b].add(a)
                
                candidates = []
                for x in range(n):
                    for y in range(x+1, n):
                        if (x, y) not in new_edges:
                            est = len(adj[x] & adj[y])
                            candidates.append((est, x, y))
                candidates.sort()
                
                for est, x, y in candidates[:10]:
                    if time.time() >= deadline:
                        break
                    cand_edge = (x, y)
                    test_edges = new_edges | {cand_edge}
                    test_books = compute_books(test_edges)
                    test_score = max_book(test_books)
                    if test_score < max_b:
                        edges = test_edges
                        books = test_books
                        improved = True
                        if test_score < best_score:
                            best_score = test_score
                            best_edges = set(edges)
                        break
                if improved:
                    break
            
            if not improved:
                edge_list = list(edges)
                random.shuffle(edge_list)
                for e in edge_list[:3]:
                    edges.discard(e)
                added = 0
                attempts = 0
                while added < 3 and attempts < 100:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x != y:
                        cand = (min(x,y), max(x,y))
                        if cand not in edges:
                            edges.add(cand)
                            added += 1
                    attempts += 1
                books = compute_books(edges)
                score = max_book(books)
                if score < best_score:
                    best_score = score
                    best_edges = set(edges)
    
    return [list(e) for e in best_edges]
