import time
import numpy as np
import random

def construct():
    n = 40
    deadline = 115.0  # Safety margin from 120s limit
    start_time = time.time()
    
    # Helper to check if adding edge (u, v) creates a C6
    # A C6 is u-v-...-u with 5 intermediate edges.
    # If we add (u, v), we need to check if there is a path of length 5 between u and v in G \ {(u,v)}.
    # Actually, simpler: maintain A5 (number of walks of length 5).
    # If A5[u, v] > 0, there is a walk of length 5. This might not be a simple path, but it's a necessary condition for a C6.
    # Wait, A5[u,v] counts walks. A walk of length 5 between u and v implies a C6 if the walk is simple.
    # However, if A5[u,v] > 0, we cannot simply say it's unsafe.
    # Standard approach: Check if there exists a path of length 5.
    # For efficiency, we often use the matrix power as a proxy or a necessary condition.
    # If A5[u,v] == 0, it is definitely safe.
    # If A5[u,v] > 0, it might be unsafe.
    # Given the density, A5[u,v] will be > 0 for most pairs.
    # A more robust check for "safe to add" in sparse-ish graphs:
    # Actually, let's stick to the matrix power method but optimize the update.
    # Or, use a direct path check if matrix update is too slow?
    # 40x40 matrix multiplication is very fast.
    
    def compute_A5(mat):
        A2 = mat @ mat
        A3 = A2 @ mat
        A4 = A3 @ mat
        A5 = A4 @ mat
        return A5

    def is_safe_to_add(A, A5, u, v):
        # If there is no walk of length 5, it's safe.
        if A5[u, v] == 0:
            return True
        # If there is a walk, we need to be careful.
        # In dense graphs, this check is too weak (false positives).
        # However, for the initial construction and early SA steps, the graph is sparse enough.
        # As the graph gets denser, we might need a more precise check.
        # For now, let's assume if A5[u,v] > 0, it's unsafe. This is conservative.
        return False

    # Initialize with a random sparse graph or disjoint K5s
    A = np.zeros((n, n), dtype=np.int64)
    # Start with 8 disjoint K5s to have a decent baseline (100 edges)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                A[u, v] = 1
                A[v, u] = 1
    
    A5 = compute_A5(A)
    current_edges = np.sum(A) // 2
    best_A = A.copy()
    best_edges = current_edges
    
    # Simulated Annealing Parameters
    T = 1000.0  # Initial temperature
    cooling_rate = 0.999  # Cooling rate per step
    min_T = 0.1
    
    # Precompute all possible edges for random selection
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() < deadline:
        if T < min_T:
            # Reheat or restart? Let's just stop or restart with a new random graph if stuck.
            # For simplicity, let's just break and return best.
            # Or, restart SA with a new random seed.
            break
        
        # Generate a 2-opt move: remove one edge, add one edge
        # Pick a random edge to remove
        if current_edges > 0:
            # Find a random edge
            # Efficient way: pick random u, v, check if edge exists. If not, try again or pick from list.
            # Since we don't maintain a dynamic list of edges, we can use a rejection sampling or maintain a list.
            # Let's maintain a list of current edges for efficiency in picking removals.
            # But updating the list on every move is O(E).
            # Alternative: Pick random u, v. If A[u,v]==1, remove. Else, pick random non-edge to add?
            # A 2-opt is typically: remove e1, add e2.
            
            # Let's pick a random edge to remove
            rem_u, rem_v = random.choice(all_edges)
            if A[rem_u, rem_v] == 0:
                continue # Not an edge, skip this iteration or retry
            
            # Remove it
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A5 = compute_A5(A)
            current_edges -= 1
            
            # Pick a random non-edge to add
            add_u, add_v = random.choice(all_edges)
            if A[add_u, add_v] == 1:
                # It became an edge? No, we just removed one. Unless rem == add.
                # If rem == add, we effectively did nothing.
                # Let's ensure add is not rem.
                if add_u == rem_u and add_v == rem_v:
                    # Revert and skip
                    A[rem_u, rem_v] = 1
                    A[rem_v, rem_u] = 1
                    A5 = compute_A5(A)
                    current_edges += 1
                    continue
                # If add was already an edge (and not the one removed), we can't add it.
                # We need a non-edge.
                # Rejection sampling for non-edge might be slow if graph is dense.
                # But we are aiming for ~105 edges out of 780, so it's sparse enough.
                # Let's just check if it's safe. If it's already an edge, it's not a valid "add" move.
                # So we need to pick a non-edge.
                # Let's try a few times to find a non-edge.
                added = False
                for _ in range(10):
                    add_u, add_v = random.choice(all_edges)
                    if A[add_u, add_v] == 0:
                        if is_safe_to_add(A, A5, add_u, add_v):
                            A[add_u, add_v] = 1
                            A[add_v, add_u] = 1
                            A5 = compute_A5(A)
                            current_edges += 1
                            added = True
                        break # Only try one add candidate per remove? 
                        # Actually, a 2-opt should be a single swap. If the swap is invalid (creates C6), we reject it.
                        # If the edge to add is already present, it's not a valid swap.
                        # So we should pick a non-edge first.
                
                if not added:
                    # Revert the removal
                    A[rem_u, rem_v] = 1
                    A[rem_v, rem_u] = 1
                    A5 = compute_A5(A)
                    current_edges += 1
                    T *= cooling_rate
                    continue
            
            # Now we have performed a swap: removed (rem_u, rem_v), added (add_u, add_v)
            # Check if this improved the edge count?
            # A 2-opt swap keeps the number of edges constant!
            # Wait, the goal is to MAXIMIZE edges.
            # A 2-opt move (remove 1, add 1) does not change the number of edges.
            # It only changes the structure.
            # This is useful for escaping local optima where we are stuck with a certain number of edges.
            # But to increase edges, we need 1-opt moves (add 1) or 2-opt moves that allow adding more later.
            # The inspiration mentioned "1-swaps" (remove 1, add 1) for SA.
            # If we only do 1-swaps, the edge count stays constant.
            # We need to allow adding edges.
            
            # Let's modify the move:
            # With probability p, do a 1-opt (add an edge).
            # With probability 1-p, do a 2-opt (remove one, add one).
            
            # Let's restart the loop logic for clarity.
            pass # The above logic was getting messy. Let's rewrite the SA loop properly.
        
        T *= cooling_rate

    # Rewrite the main SA loop properly
    # Reset state
    A = np.zeros((n, n), dtype=np.int64)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                A[u, v] = 1
                A[v, u] = 1
    A5 = compute_A5(A)
    current_edges = np.sum(A) // 2
    best_A = A.copy()
    best_edges = current_edges
    
    T = 500.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    # Precompute list of edges and non-edges? 
    # Maintaining lists is expensive. 
    # Let's use random sampling with rejection.
    
    while time.time() < deadline and T > min_T:
        
        # Decide move type
        if random.random() < 0.5:
            # 1-opt: Add an edge
            # Try to add a random non-edge
            for _ in range(5): # Try a few times
                u, v = random.choice(all_edges)
                if A[u, v] == 0:
                    if is_safe_to_add(A, A5, u, v):
                        # Accept if it improves (it always does, +1 edge)
                        A[u, v] = 1
                        A[v, u] = 1
                        A5 = compute_A5(A)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_A = A.copy()
                    break
        else:
            # 2-opt: Remove one edge, add one edge
            # Pick random edge to remove
            rem_u, rem_v = random.choice(all_edges)
            if A[rem_u, rem_v] == 0:
                continue
            
            # Remove it
            A[rem_u, rem_v] = 0
            A[rem_v, rem_u] = 0
            A5 = compute_A5(A)
            current_edges -= 1
            
            # Pick random non-edge to add
            add_u, add_v = random.choice(all_edges)
            if A[add_u, add_v] == 1:
                # Invalid move, revert
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A5 = compute_A5(A)
                current_edges += 1
                T *= cooling_rate
                continue
            
            if is_safe_to_add(A, A5, add_u, add_v):
                # The swap is valid. Edge count is unchanged.
                # In SA, we accept worse moves with probability exp(-delta/T).
                # Here delta = 0 (same edge count).
                # But we might be stuck. 
                # Actually, if we just swap, we don't gain edges.
                # We need to eventually add edges.
                # This 2-opt is for "mixing" to find better configurations.
                A[add_u, add_v] = 1
                A[add_v, add_u] = 1
                A5 = compute_A5(A)
                current_edges += 1 # Back to original count
            else:
                # Swap creates C6, revert
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A5 = compute_A5(A)
                current_edges += 1
            
        T *= cooling_rate

    # Final greedy polish
    # Try to add any remaining safe edges
    A5 = compute_A5(best_A)
    for u, v in all_edges:
        if best_A[u, v] == 0 and A5[u, v] == 0:
            best_A[u, v] = 1
            best_A[v, u] = 1
            A5 = compute_A5(best_A)

    edges_out = []
    for u, v in all_edges:
        if best_A[u, v] == 1:
            edges_out.append([u, v])
    return edges_out
