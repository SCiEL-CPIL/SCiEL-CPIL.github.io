                separated = False
                min_pen = float('inf')
                for k in range(6):
                    if max_p_i[k] < min_p_j[k] or max_p_j[k] < min_p_i[k]:
                        separated = True
                        break
                    pen = min(max_p_i[k], max_p_j[k]) - max(min_p_i[k], min_p_j[k])
                    if pen < min_pen:
                        min_pen = pen
                if not separated:
                    penalty += min_pen
