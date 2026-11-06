# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    rng = random.Random(42)
    
    # Start with the Pinwheel construction
    # a: cycle (i -> i+1), b: 0->0, i->i-1
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] + [(i - 1) % N for i in range(1, N)]
    best_len = _compute_reset_length(best_a, best_b)
    
    if best_len < 0:
        # Fallback if pinwheel fails (should not happen)
        return [best_a, best_b]

    current_a = list(best_a)
    current_b = list(best_b)
    current_len = best_len
    
    def _local_search(a, b, curr_len):
        """
        Greedy local search: try all single swaps and single changes.
        Returns (new_a, new_b, new_len, improved)
        """
        improved = False
        best_move_a = list(a)
        best_move_b = list(b)
        best_move_len = curr_len
        
        # 1. Try single swaps in a
        for i in range(N):
            for j in range(i+1, N):
                a_new = list(a)
                a_new[i], a_new[j] = a_new[j], a_new[i]
                # Quick check: is it synchronizing?
                if _is_synchronizing(a_new, b):
                    l = _compute_reset_length(a_new, b)
                    if l > best_move_len:
                        best_move_len = l
                        best_move_a = a_new
                        best_move_b = list(b)
                        improved = True
        
        # 2. Try single swaps in b
        for i in range(N):
            for j in range(i+1, N):
                b_new = list(b)
                b_new[i], b_new[j] = b_new[j], b_new[i]
                if _is_synchronizing(a, b_new):
                    l = _compute_reset_length(a, b_new)
                    if l > best_move_len:
                        best_move_len = l
                        best_move_a = list(a)
                        best_move_b = b_new
                        improved = True

        # 3. Try single changes in a
        for i in range(N):
            for val in range(N):
                if val == a[i]:
                    continue
                a_new = list(a)
                a_new[i] = val
                if _is_synchronizing(a_new, b):
                    l = _compute_reset_length(a_new, b)
                    if l > best_move_len:
                        best_move_len = l
                        best_move_a = a_new
                        best_move_b = list(b)
                        improved = True

        # 4. Try single changes in b
        for i in range(N):
            for val in range(N):
                if val == b[i]:
                    continue
                b_new = list(b)
                b_new[i] = val
                if _is_synchronizing(a, b_new):
                    l = _compute_reset_length(a, b_new)
                    if l > best_move_len:
                        best_move_len = l
                        best_move_a = list(a)
                        best_move_b = b_new
                        improved = True

        return best_move_a, best_move_b, best_move_len, improved

    # Main loop: Local Search + Restarts
    while time.time() < deadline - 5.0:
        # Run local search to convergence
        a_ls, b_ls, len_ls, improved = _local_search(current_a, current_b, current_len)
        
        if improved:
            current_a, current_b, current_len = a_ls, b_ls, len_ls
            if current_len > best_len:
                best_a, best_b, best_len = list(current_a), list(current_b), current_len
        else:
            # Stuck at local optimum. Perturb and restart.
            # Perturb by randomizing a few positions
            perturb_count = rng.randint(3, 6)
            current_a = list(best_a)
            current_b = list(best_b)
            current_len = best_len
            
            for _ in range(perturb_count):
                if rng.random() < 0.5:
                    pos = rng.randrange(N)
                    current_a[pos] = rng.randrange(N)
                else:
                    pos = rng.randrange(N)
                    current_b[pos] = rng.randrange(N)
            
            # Check if perturbed state is even synchronizing
            if not _is_synchronizing(current_a, current_b):
                # If not, revert to best and try again with different perturbation
                # Or just continue loop to try another perturbation
                continue
                
            current_len = _compute_reset_length(current_a, current_b)
            if current_len < 0:
                continue

    return [best_a, best_b]
# EVOLVE-BLOCK-END
