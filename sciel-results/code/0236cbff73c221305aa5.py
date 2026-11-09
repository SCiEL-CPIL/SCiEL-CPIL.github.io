            best_edge = -1
            max_cov = -1
            candidates = []
            for e in range(num_edges):
                if e in removed:
                    continue
                if edge_cover_count[e] > max_cov:
                    max_cov = edge_cover_count[e]
                    candidates = [e]
                elif edge_cover_count[e] == max_cov:
                    candidates.append(e)
            if candidates:
                best_edge = random.choice(candidates)
