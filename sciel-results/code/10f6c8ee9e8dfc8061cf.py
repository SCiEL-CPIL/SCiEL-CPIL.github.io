    def smart_greedy(masks):
        count_before = count_edges(masks)
        improved = True
        swaps = 0
        while improved and time.time() < deadline - 5:
            improved = False
            non_edges = get_non_edges(masks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if (masks[u] >> v) & 1:
                    continue
                add_edge(masks, u, v)
                if is_valid(masks):
                    improved = True
                else:
                    viol = find_violation(masks)
                    if viol:
                        i, j, k, common = viol
                        bits = [b for b in range(n) if (common >> b) & 1]
                        v_common = random.choice(bits)
                        v_trip = random.choice([i, j, k])
                        remove_edge(masks, v_trip, v_common)
                        swaps += 1
                        if swaps > 100: # Limit swaps per pass
                            break
                        improved = True # Continue searching
                    else:
                        remove_edge(masks, u, v)
        
        return count_edges(masks) > count_before
