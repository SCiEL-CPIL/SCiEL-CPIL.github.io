import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses Simulated Annealing with a bitset-accelerated C6 checker.
    """
    N = 40
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit

    # Precompute all pairs
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    pair_index = {}
    for idx, (u, v) in enumerate(all_pairs):
        pair_index[(u, v)] = idx
        pair_index[(v, u)] = idx

    # Helper to check if adding edge (u, v) creates a C6.
    def can_add(adj_bits, u, v):
        start_mask = 1 << u
        stack = [(u, 0, start_mask)]
        
        while stack:
            node, depth, mask = stack.pop()
            
            if depth == 4:
                if adj_bits[node] & (1 << v):
                    return False
                continue
            
            neighbors = adj_bits[node]
            neighbors &= ~(1 << v)
            neighbors &= ~mask
            
            while neighbors:
                nb = (neighbors & -neighbors).bit_length() - 1
                neighbors &= neighbors - 1
                stack.append((nb, depth + 1, mask | (1 << nb)))
        
        return True

    def edge_count(bits):
        return sum(bin(b).count('1') for b in bits) // 2

    # Initialize with 8 disjoint K5s
    adj_bits = [0] * N
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
    current_edges = edge_count(adj_bits)
    best_edges = current_edges
    best_adj_bits = adj_bits[:]

    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    num_steps = 0
    max_steps = 500000  # Will be limited by time

    while time.time() < deadline:
        if T < T_min:
            # Restart with higher temperature if stuck for too long
            T = 0.5
            # Reset best if we've been stuck? No, keep best.
            # Maybe try a kick to escape local minimum
            if random.random() < 0.5:
                # Remove some random edges
                existing_edges = []
                for u in range(N):
                    for v in range(u + 1, N):
                        if adj_bits[u] & (1 << v):
                            existing_edges.append((u, v))
                random.shuffle(existing_edges)
                for _ in range(random.randint(2, 5)):
                    if not existing_edges: break
                    u, v = existing_edges.pop()
                    adj_bits[u] &= ~(1 << v)
                    adj_bits[v] &= ~(1 << u)
                    current_edges -= 1
                T = 1.0

        # Choose a move: add or remove an edge
        if random.random() < 0.6:
            # Try to add an edge
            # Pick a random missing edge
            # To make this efficient, we can just pick a random pair and check if it's missing
            # Or maintain a list of missing edges. Let's maintain a list for efficiency.
            # But maintaining a list is O(N^2) to update.
            # Given N=40, N^2=1600. It's small.
            
            # Let's just pick a random pair.
            while True:
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u == v: continue
                if u > v: u, v = v, u
                if not (adj_bits[u] & (1 << v)):
                    break
            
            if can_add(adj_bits, u, v):
                # Add edge
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                new_edges = current_edges + 1
                # Always accept improvement
                current_edges = new_edges
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj_bits = adj_bits[:]
            else:
                # Reject
                pass
        else:
            # Try to remove an edge
            # Pick a random existing edge
            while True:
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u == v: continue
                if u > v: u, v = v, u
                if adj_bits[u] & (1 << v):
                    break
            
            # Remove edge
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            new_edges = current_edges - 1
            delta = -1
            
            # Accept with probability exp(delta / T)
            if random.random() < np.exp(delta / T):
                current_edges = new_edges
            else:
                # Reject, put edge back
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)

        T *= alpha
        num_steps += 1

    # Final polish: try to add any possible edges greedily
    for u, v in all_pairs:
        if not (best_adj_bits[u] & (1 << v)):
            if can_add(best_adj_bits, u, v):
                best_adj_bits[u] |= (1 << v)
                best_adj_bits[v] |= (1 << u)
                best_edges += 1

    # Return best graph
    final_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj_bits[u] & (1 << v):
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
