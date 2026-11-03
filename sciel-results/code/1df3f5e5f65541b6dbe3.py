            # Inside local_search_2opt
            stuck_count = 0
            # ...
            if best_delta < -1e-9:
                # ... apply swap
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > 5 and time.time() < time_limit:
                    # Random kick
                    i, j = np.random.choice(n, 2, replace=False)
                    p[i], p[j] = p[j], p[i]
                    current_cost = get_cost(p)
                    stuck_count = 0
                else:
                    break
