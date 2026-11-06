    # 3. Polish
    if best_solution:
        A = best_solution.copy()
        improved = True
        while improved:
            improved = False
            for i in range(len(A)):
                if A[i] == 0: continue
                old = A[i]
                # Try to reduce A[i]
                for new_val in range(1, old):
                    if new_val in A: continue
                    trial = A.copy()
                    trial[i] = new_val
                    trial = sorted(trial)
                    if is_valid(trial):
                        A = trial
                        improved = True
                        break
        update_best(A)
