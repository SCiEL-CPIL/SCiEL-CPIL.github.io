    def heuristic_init():
        S = set(range(1, N + 1))
        # Count conflicts for each element
        # An element x is in a bad triple if there exist y,z in S such that (x,y,z) is bad.
        # We want to remove elements that are in the most bad triples.
        # Iterate until no bad triples.
        changed = True
        while changed:
            changed = False
            # Find all bad triples in S
            bads = []
            S_list = sorted(S)
            for i in range(len(S_list)):
                a = S_list[i]
                for j in range(i+1, len(S_list)):
                    b = S_list[j]
                    lab = lcm_table[(a,b)]
                    for k in range(j+1, len(S_list)):
                        c = S_list[k]
                        if lcm_table[(a,c)] == lab and lcm_table[(b,c)] == lab:
                            bads.append((a,b,c))
            if not bads:
                break
            # Count frequency of each element in bads
            freq = {}
            for (a,b,c) in bads:
                freq[a] = freq.get(a, 0) + 1
                freq[b] = freq.get(b, 0) + 1
                freq[c] = freq.get(c, 0) + 1
            # Remove element with max freq
            # Tie-break by value (smaller value first? or larger? usually smaller values are more "central" in LCM structures)
            max_freq = max(freq.values())
            candidates = [x for x in freq if freq[x] == max_freq]
            to_remove = min(candidates) # Arbitrary tie break
            S.remove(to_remove)
            changed = True
        return S
    