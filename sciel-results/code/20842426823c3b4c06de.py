<<<<<<< SEARCH
    # Fallback: Greedy construction similar to seed
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
=======
    # Fallback: Greedy construction with minimal colors
    # Use a smarter greedy that tries to reuse existing colors
    colors = []
    for x in range(1, 129):
        # Try existing colors first, then new colors
        used = set(colors)
        candidate_colors = list(used) + list(range(len(used), 128))
        for color in candidate_colors:
            valid = True
            for d in range(1, (x-1)//3 + 1):
                if x - 1 - 3*d >= 0:
                    ap_colors = {colors[x-1-d], colors[x-1-2*d], colors[x-1-3*d]}
                    if len({color} | ap_colors) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
    return colors
>>>>>>> REPLACE
