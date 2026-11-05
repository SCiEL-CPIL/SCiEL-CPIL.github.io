<<<<<<< SEARCH
    def local_search_swaps(S, remaining):
        """Try swaps: remove one element from S, add one from remaining."""
        improved = False
        s_list = list(S)
        rem_list = list(remaining)
        for a in s_list:
            if time.time() > deadline - 2:
                break
            S_minus_a = S - {a}
            for x in rem_list:
                if x in S:
                    continue
                if can_add_fast(x, S_minus_a, None):
                    S.add(x)
                    S.discard(a)
                    improved = True
                    return S, True
        return S, improved
=======
    def local_search_swaps(S, remaining):
        """Try swaps: remove one element from S, add one from remaining.
        Tries multiple swaps per pass to escape local optima."""
        s_list = list(S)
        rem_list = list(remaining)
        swaps_made = 0
        max_swaps_per_pass = 10
        for a in s_list:
            if time.time() > deadline - 2 or swaps_made >= max_swaps_per_pass:
                break
            S_minus_a = S - {a}
            for x in rem_list:
                if x in S:
                    continue
                if can_add_fast(x, S_minus_a, None):
                    S.add(x)
                    S.discard(a)
                    swaps_made += 1
                    if swaps_made >= max_swaps_per_pass:
                        break
        return S, swaps_made > 0
>>>>>>> REPLACE

