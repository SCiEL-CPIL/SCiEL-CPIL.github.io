    else:
        cross = dx * (py - y1) - dy * (px - x1)
        dist = abs(cross) / math.sqrt(length_sq)
        return -dist if cross < 0 else dist
