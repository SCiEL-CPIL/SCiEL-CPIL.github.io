            # Perturb
            current_set = set(best_set)
            # Remove 1 or 2 random elements
            k = random.randint(1, 2)
            to_remove = random.sample(list(current_set), min(k, len(current_set)))
            for r in to_remove:
                current_set.remove(r)
