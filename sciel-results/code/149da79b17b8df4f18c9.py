                if random.random() < 0.33:
                    best_set = list(range(1, n+1))
                elif random.random() < 0.5:
                    best_set = list(range(2, 2*n+1, 2))
                else:
                    best_set = [random.randint(1, search_limit) for _ in range(n)]
                    while len(set(best_set)) < n:
                        best_set[random.randint(0, n-1)] = random.randint(1, search_limit)
