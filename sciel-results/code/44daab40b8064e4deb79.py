        max_v = -1
        for e in best_edges:
            for v in e:
                if v > max_v:
                    max_v = v
        n = max_v + 1
        return {'n': n, 'edges': best_edges}
