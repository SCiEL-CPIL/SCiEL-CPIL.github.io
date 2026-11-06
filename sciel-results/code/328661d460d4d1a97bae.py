import numpy as np
import random
import time

def construct():
    n = 20
    best_edges = -1
    best_A = None
    deadline = time.time() + 115

    def is_valid(A):
        """Check no K2,3: no two vertices have >=3 common neighbors."""
        K = A @ A
        for u in range(n):
            for v in range(u + 1, n):
                if K[u, v] >= 3:
                    return False
        return True

    def edge_count(A):
        return int(np.sum(A)) // 2

    def can_add_edge(A, K, u, v):
        """Check if edge (u,v) can be added without creating K2,3."""
        if A[u, v] == 1:
            return False
        # Adding (u,v): u becomes common neighbor of x and v for x in N(u)\{v}
        # v becomes common neighbor of u and y for y in N(v)\{u}
        for x in range(n):
            if A[u, x] == 1 and x != v:
                if K[x, v] >= 2:
                    return False
        for y in range(n):
            if A[v, y] == 1 and y != u:
                if K[u, y] >= 2:
                    return False
        return True

    def add_edge(A, K, u, v):
        """Add edge (u,v) and update K."""
        A[u, v] = 1
        A[v, u] = 1
        for x in range(n):
            if A[u, x] == 1 and x != v:
                K[x, v] += 1
                K[v, x] += 1
        for y in range(n):
            if A[v, y] == 1 and y != u:
                K[u, y] += 1
                K[y, u] += 1

    def remove_edge(A, K, u, v):
        """Remove edge (u,v) and update K."""
        A[u, v] = 0
        A[v, u] = 0
        for x in range(n):
            if A[u, x] == 1 and x != v:
                K[x, v] -= 1
                K[v, x] -= 1
        for y in range(n):
            if A[v, y] == 1 and y != u:
                K[u, y] -= 1
                K[y, u] -= 1

    def try_add_edges(A, K, count):
        """Greedily add edges while maintaining validity."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for u in range(n):
                if time.time() >= deadline:
                    return A, K, count
                for v in range(u + 1, n):
                    if A[u, v] == 0 and can_add_edge(A, K, u, v):
                        add_edge(A, K, u, v)
                        count += 1
                        changed = True
                        break
        return A, K, count

    def remove_to_fix(A, K, count):
        """Remove edges until valid."""
        while time.time() < deadline:
            viol = False
            for u in range(n):
                if time.time() >= deadline:
                    break
                for v in range(u + 1, n):
                    if K[u, v] >= 3:
                        viol = True
                        zs = [z for z in range(n) if A[u, z] == 1 and A[v, z] == 1]
                        random.shuffle(zs)
                        for z in zs:
                            remove_edge(A, K, u, z)
                            count -= 1
                            break
                        break
                if viol:
                    break
            if not viol:
                break
        return A, K, count

    def try_swap_edges(A, K, count):
        """Try edge swaps to improve the graph."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing an edge and adding another
            for u1 in range(n):
                if time.time() >= deadline:
                    return A, K, count
                for v1 in range(u1 + 1, n):
                    if A[u1, v1] == 1:
                        # Remove edge (u1, v1)
                        remove_edge(A, K, u1, v1)
                        count -= 1
                        
                        # Try adding various edges
                        for u2 in range(n):
                            if time.time() >= deadline:
                                break
                            for v2 in range(u2 + 1, n):
                                if A[u2, v2] == 0 and can_add_edge(A, K, u2, v2):
                                    # Check if this swap improves things
                                    # For now, just add if possible
                                    add_edge(A, K, u2, v2)
                                    count += 1
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                        else:
                            # Revert
                            add_edge(A, K, u1, v1)
                            count += 1
                    if improved:
                        break
                if improved:
                    break
        return A, K, count

    def simulated_annealing(A, K, count, steps=1000):
        """Simulated annealing to escape local optima."""
        temp = 1.0
        cooling = 0.999
        for step in range(steps):
            if time.time() >= deadline:
                break
            # Random move: remove an edge
            edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 1]
            if not edges_list:
                break
            u1, v1 = random.choice(edges_list)
            remove_edge(A, K, u1, v1)
            count -= 1
            
            # Try to add a random edge
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            if non_edges:
                u2, v2 = random.choice(non_edges)
                if can_add_edge(A, K, u2, v2):
                    add_edge(A, K, u2, v2)
                    count += 1
                else:
                    # Revert
                    add_edge(A, K, u1, v1)
                    count += 1
            
            # Greedy add
            A, K, count = try_add_edges(A, K, count)
            temp *= cooling
        return A, K, count

    # Strategy: Start with 2 disjoint K10s, then optimize
    for restart in range(2000):
        if time.time() >= deadline:
            break
        
        # Start with 2 disjoint K10s
        A = np.zeros((n, n), dtype=int)
        for a in range(10):
            for b in range(a + 1, 10):
                A[a, b] = 1
                A[b, a] = 1
        for a in range(10, 20):
            for b in range(a + 1, 20):
                A[a, b] = 1
                A[b, a] = 1
        
        # Randomly permute vertices
        perm = list(range(n))
        random.shuffle(perm)
        A2 = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(n):
                if A[u, v] == 1:
                    A2[perm[u], perm[v]] = 1
        A = A2
        K = A @ A
        count = 90
        
        # Repair and optimize
        A, K, count = remove_to_fix(A, K, count)
        A, K, count = try_add_edges(A, K, count)
        
        # Try swaps
        A, K, count = try_swap_edges(A, K, count)
        
        # Simulated annealing
        if time.time() < deadline - 30:
            A, K, count = simulated_annealing(A, K, count, steps=500)
        
        if count > best_edges:
            best_edges = count
            best_A = A.copy()
        
        # Early stop if we're doing well
        if best_edges >= 120 and time.time() > deadline - 20:
            break

    # Fallback
    if best_A is None:
        best_A = np.zeros((n, n), dtype=int)
        for k in range(5):
            for a in range(4):
                for b in range(a + 1, 4):
                    u = 4 * k + a
                    v = 4 * k + b
                    best_A[u, v] = 1
                    best_A[v, u] = 1
        best_edges = 30

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges
