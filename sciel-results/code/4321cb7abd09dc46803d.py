def construct():
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a,b], [a,c], [b,c]]
    return {'n': 30, 'edges': edges}
