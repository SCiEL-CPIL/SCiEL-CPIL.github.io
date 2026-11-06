import random
import time

def construct():
        n = 20
        deadline = time.time() + 110
        best_count = 0
        best_masks = [0] * n

        def count_edges(masks):
            return sum(m.bit_count() for m in masks) // 2

        def is_valid(masks):
            for i in range(n):
                mi = masks[i]
                for j in range(i + 1, n):
                    mij = mi & masks[j]
                    for k in range(j + 1, n):
                        if (mij & masks[k]).bit_count() >= 3:
                            return False
            return True

        def find_violation(masks):
            for i in range(n):
                mi = masks[i]
                for j in range(i + 1, n):
                    mij = mi & masks[j]
                    for k in range(j + 1, n):
                        common = mij & masks[k]
                        if common.bit_count() >= 3:
                            return (i, j, k, common)
            return None

        def add_edge(masks, u, v):
            masks[u] |= 1 << v
            masks[v] |= 1 << u

        def remove_edge(masks, u, v):
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)

        def greedy_add(masks):
            # Try to add all missing edges
            improved = True
            while improved:
                improved = False
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (masks[u] >> v) & 1:
                            # Try add
                            add_edge(masks, u, v)
                            if is_valid(masks):
                                improved = True
                            else:
                                remove_edge(masks, u, v)

        def repair(masks):
            while True:
                viol = find_violation(masks)
                if viol is None:
                    break
                i, j, k, common = viol
                # Remove a random edge from the K3,3
                # Edges are between {i,j,k} and bits in common
                # Pick a random bit from common
                bits = [b for b in range(n) if (common >> b) & 1]
                # Actually, any edge in the K3,3 works.
                # Let's pick a random vertex from common and a random vertex from {i,j,k}
                v_common = random.choice(bits)
                v_trip = random.choice([i, j, k])
                remove_edge(masks, v_trip, v_common)

        def local_search(masks):
            while time.time() < deadline - 5:
                # Try to add a random non-edge
                # Find all non-edges
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if not (masks[u] >> v) & 1:
                            non_edges.append((u, v))
                
                if not non_edges:
                    break # Complete graph? Unlikely, or local optimum with no missing edges? No, if no missing edges, it's complete.
                
                # If many non-edges, pick random
                u, v = random.choice(non_edges)
                
                add_edge(masks, u, v)
                if is_valid(masks):
                    continue # Success, loop to add more
                
                # Invalid, revert
                remove_edge(masks, u, v)
                
                # Perturb: Remove 2-4 random edges to break local optimum
                num_remove = random.randint(2, 4)
                current_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if (masks[u] >> v) & 1:
                            current_edges.append((u, v))
                
                if len(current_edges) <= num_remove:
                    # Too sparse, just re-greedy
                    greedy_add(masks)
                    continue

                for _ in range(num_remove):
                    e = random.choice(current_edges)
                    remove_edge(masks, e[0], e[1])
                    current_edges.remove(e)
                
                # Re-greedy to fill
                greedy_add(masks)

        # Main loop
        for trial in range(200):
            if time.time() > deadline - 10:
                break
            
            # Random init
            masks = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        add_edge(masks, u, v)
            
            repair(masks)
            greedy_add(masks)
            
            current_count = count_edges(masks)
            if current_count > best_count:
                best_count = current_count
                best_masks = masks[:]
            
            # Local search
            local_search(masks)
            
            current_count = count_edges(masks)
            if current_count > best_count:
                best_count = current_count
                best_masks = masks[:]

        # Convert best_masks to list of edges
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if (best_masks[u] >> v) & 1:
                    result.append([u, v])
        return result
