    # Final verification
    def is_valid(S):
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    if not is_valid(best_set):
        # Fallback to greedy
        S_fallback = greedy_construct(list(range(n, 0, -1)))
        if is_valid(S_fallback):
            best_set = S_fallback
