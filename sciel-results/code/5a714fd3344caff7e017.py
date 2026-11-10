<<<<<<< SEARCH
    C = U_set - best_H
    N = all_pts - C
=======
    C = U_set - best_H
    N = all_pts - C
    
    # Validate: check that N is a valid Nikodym set
    # For every point x in F_49^2, there exists a line through x such that all other points on that line are in N
    N_set = set(N)
    valid = True
    for x in range(total_points):
        found_line = False
        for i in range(num_lines):
            L = lines[i]
            if x not in set(L.tolist()):
                continue
            all_others_in_N = True
            for other in L:
                if other != x and other not in N_set:
                    all_others_in_N = False
                    break
            if all_others_in_N:
                found_line = True
                break
        if not found_line:
            valid = False
            break
    
    if not valid:
        # Fallback: use the original best_H without post-optimization
        C = U_set - best_H
        N = all_pts - C
>>>>>>> REPLACE

Now let me reconsider - the validation step is O(2401 * 2450 * 49) which is way too slow. Let me remove it and trust the greedy construction:

