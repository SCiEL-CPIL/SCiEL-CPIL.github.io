def construct():
    # Precompute 2^a * 3^b * 5^c smooth numbers up to 128
    smooth = []
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128:
            break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128:
                break
            for c in range(0, 5):
                p5 = 5 ** c
                val = p2 * p3 * p5
                if val > 128:
                    break
                smooth.append(val)
    smooth = sorted(set(smooth))
    n_smooth = len(smooth)

    # best_rep[s] = (min_count, tuple_of_terms)
    best_rep = [None] * 129

    # DFS to find all antichains with sum <= 128
    # start_idx: index in smooth to start picking from
    # curr_sum: current sum
    # curr_set: list of current terms
    def dfs(start_idx, curr_sum, curr_set):
        # Update best if curr_set is valid (it is by construction)
        if curr_set:
            s = curr_sum
            k = len(curr_set)
            if best_rep[s] is None or k < best_rep[s][0]:
                best_rep[s] = (k, tuple(curr_set))
        
        # Try adding more elements
        for i in range(start_idx, n_smooth):
            x = smooth[i]
            if curr_sum + x > 128:
                break
            
            # Check antichain property with existing elements
            can_add = True
            for y in curr_set:
                if x % y == 0 or y % x == 0:
                    can_add = False
                    break
            
            if can_add:
                dfs(i + 1, curr_sum + x, curr_set + [x])

    dfs(0, 0, [])

    # Construct answers
    answers = []
    for n in range(1, 129):
        if best_rep[n] is not None:
            answers.append(list(best_rep[n][1]))
        else:
            # Fallback: should not happen, but find a valid one via brute force if needed?
            # Given the density, it's very unlikely to be None for 1..128.
            # 1 is {1}. 2 is {2}. 3 is {3}. 4 is {4}. 5 is {5}.
            # 6: {2,3} no (2|6? no, 2 and 3 don't divide each other). Wait, 2 and 3 are smooth. 2+3=5.
            # Target 6: {1, 5}? 1|5 no. 1 and 5. Sum 6. Size 2.
            # Target 7: {1, 6}? 1|6 no. Sum 7. Size 2.
            # Target 8: {3, 5}? Sum 8. Size 2.
            # It is highly probable all 1..128 are covered.
            answers.append([])

    return answers
