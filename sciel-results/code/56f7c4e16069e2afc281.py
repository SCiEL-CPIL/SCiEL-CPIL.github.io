import random
import time

def construct():
    deadline = time.time() + 140
    n = 40
    target_edges = 401

    random.seed(42)

    def popcount(x):
        return bin(x).count('1')

    def compute_books(adj_bits, edge_set):
        books = {}
        for u, v in edge_set:
            common = adj_bits[u] & adj_bits[v]
            cnt = popcount(common)
            books[(u, v)] = cnt
        return books

    # Initialize with random graph
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = set(random.sample(all_possible, target_edges))

    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    books = compute_books(adj_bits, edges)
    best_max = max(books.values())
    best_edges = set(edges)

    # Local search with edge swaps
    while time.time() < deadline:
        cur_max = max(books.values())
        if cur_max <= 7:
            break

        # Focus on edges with high book sizes
        worst_edges = [(e, b) for e, b in books.items() if b >= cur_max - 1]
        random.shuffle(worst_edges)

        improved = False
        for (u, v), _ in worst_edges[:8]:
            if time.time() >= deadline:
                break

            for _ in range(300):
                if time.time() >= deadline:
                    break
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                if (x, y) in edges or (y, x) in edges:
                    continue

                new_edges = set(edges)
                new_edges.discard((u, v))
                new_edges.discard((v, u))
                new_edges.add((x, y))

                if len(new_edges) != target_edges:
                    continue

                new_adj_bits = list(adj_bits)
                new_adj_bits[u] &= ~(1 << v)
                new_adj_bits[v] &= ~(1 << u)
                new_adj_bits[x] |= (1 << y)
                new_adj_bits[y] |= (1 << x)

                new_books = compute_books(new_adj_bits, new_edges)
                new_max = max(new_books.values())

                if new_max < best_max:
                    best_max = new_max
                    best_edges = new_edges
                    edges = new_edges
                    adj_bits = new_adj_bits
                    books = new_books
                    improved = True
                    break

        if not improved:
            # Perturbation to escape local optima
            if random.random() < 0.7:
                random_edge = random.choice(tuple(edges))
                for _ in range(100):
                    if time.time() >= deadline:
                        break
                    x = random.randint(0, n - 1)
                    y = random.randint(0, n - 1)
                    if x == y:
                        continue
                    if (x, y) in edges or (y, x) in edges:
                        continue

                    new_edges = set(edges)
                    new_edges.discard(random_edge)
                    new_edges.discard((random_edge[1], random_edge[0]))
                    new_edges.add((x, y))

                    if len(new_edges) != target_edges:
                        continue

                    new_adj_bits = list(adj_bits)
                    a, b = random_edge
                    new_adj_bits[a] &= ~(1 << b)
                    new_adj_bits[b] &= ~(1 << a)
                    new_adj_bits[x] |= (1 << y)
                    new_adj_bits[y] |= (1 << x)

                    new_books = compute_books(new_adj_bits, new_edges)
                    new_max = max(new_books.values())

                    if new_max <= best_max:
                        if new_max < best_max:
                            best_max = new_max
                            best_edges = new_edges
                        edges = new_edges
                        adj_bits = new_adj_bits
                        books = new_books
                        break

    result = [[u, v] for u, v in best_edges]
    return result
