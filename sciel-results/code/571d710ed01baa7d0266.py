def construct():
    points = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            d2 = a*a + a*b + b*b
            points.append((d2, a, b))
    
    points.sort()
    
    selected = [ [a, b] for d2, a, b in points[:32] ]
    
    return selected
