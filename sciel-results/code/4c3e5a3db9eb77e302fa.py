    config3 = []
    config3.append([0, 0, 0])
    for i in range(3):
        config3.append([i - 1, sqrt3/2, 0])
    for i in range(3):
        config3.append([i - 1, sqrt3, 0])
    for i in range(3):
        config3.append([i - 1, 3*sqrt3/2, 0])
    config3.append([0, 2*sqrt3, 0])
    configs.append(np.array(config3))
