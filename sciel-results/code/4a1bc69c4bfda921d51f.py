        for seed in range(10):
            ...
            for x in candidates:
                ...
                if not products.intersection(tails):
                    values.append(x)
                    values.sort()
                    products.update(tails)
            if len(values) > len(best):
                best = values
      