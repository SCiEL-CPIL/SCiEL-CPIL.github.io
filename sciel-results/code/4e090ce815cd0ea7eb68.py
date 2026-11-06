import time
import random

def construct():
    n = 40
    start_time = time.time()
    deadline = 145

    def would_create_c6(adj_bits, u, v):
        Nu = adj_bits[u]
        Nv = adj_bits[v]
        w_bits = Nu
        while w_bits:
            lsb = w_bits & (-w_bits)
            w = lsb.bit_length() - 1
            w_bits ^= lsb
            if w == v:
                continue
            x_bits = Nv & ~(1 << w)
            while x_bits:
                lsb_x = x_bits & (-x_bits)
                x = lsb_x.bit_length() - 1
                x_bits ^= lsb_x
                z_bits = adj_bits[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                while z_bits:
                    lsb_z = z_bits & (-z_bits)
                    z = lsb_z.bit_length() - 1
                    z_bits ^= lsb_z
                    y_bits = adj_bits[z] & ~(1 << u) & ~(1 << v) & ~(1 << w) & ~(1 << x)
                    if y_bits & adj_bits[x]:
                        return True
        return False

    def try_greedy(seed_val):
        rng = random.Random(seed_val)
        adj_bits = [0] * n
        edges = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append([u, v])

        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        non_edges = [(u, v) for u, v in all_pairs if not (adj_bits[u] >> v) & 1]
        rng.shuffle(non_edges)

        for u, v in non_edges:
            if time.time() - start_time > deadline - 5:
                break
            if (adj_bits[u] >> v) & 1:
                continue
            if not would_create_c6(adj_bits, u, v):
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                edges.append([u, v])
        return edges

    best_edges = []
    for seed_val in range(50):
        if time.time() - start_time > deadline - 10:
            break
        edges = try_greedy(seed_val)
        if len(edges) > len(best_edges):
            best_edges = edges

    return best_edges
