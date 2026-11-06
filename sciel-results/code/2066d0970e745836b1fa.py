def construct():
    start_time = time.time()
    DEADLINE = 85.0
    N = 128
    
    aps4 = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps4.append((i, i + d, i + 2 * d, i + 3 * d))
    
    def get_forbidden_pairs(colors):
        """Get set of (c1, c2) pairs that cannot be merged."""
        forbidden = set()
        for ap in aps4:
            c_set = set(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
            if len(c_set) == 3:
                # Any pair from c_set would cause violation if merged
                c_list = sorted(c_set)
                for i in range(len(c_list)):
                    for j in range(i+1, len(c_list)):
                        forbidden.add((c_list[i], c_list[j]))
        return forbidden
    
    def greedy_merge_fast(colors):
        colors = colors[:]
        while True:
            if time.time() - start_time > DEADLINE:
                break
            forbidden = get_forbidden_pairs(colors)
            current_colors = sorted(set(colors))
            merged = False
            for i in range(len(current_colors)):
                for j in range(i+1, len(current_colors)):
                    c1, c2 = current_colors[i], current_colors[j]
                    if (c1, c2) not in forbidden:
                        # Merge c2 into c1
                        for k in range(N):
                            if colors[k] == c2:
                                colors[k] = c1
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
        return colors
