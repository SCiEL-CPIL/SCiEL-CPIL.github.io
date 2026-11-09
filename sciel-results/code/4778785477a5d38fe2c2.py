    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        calls_made = 0
        dim = self.dim
        budget = self.budget

        # Initialize simplex for Nelder-Mead
        x0 = np.zeros(dim)
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += 0.5
            simplex.append(xi)
        
        fx = []
        for x in simplex:
            fx.append(func(np.clip(x, -5, 5)))
            calls_made += 1
            
        best_x = simplex[np.argmin(fx)].copy()
        best_fx = min(fx)

        # Nelder-Mead Loop
        while calls_made < budget and time.time() < deadline:
            # Sort simplex
            order = np.argsort(fx)
            simplex = [simplex[i] for i in order]
            fx = [fx[i] for i in order]
            
            # Centroid of best dim points
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls_made += 1
            if fr < best_fx:
                best_fx = fr
                best_x = xr.copy()
                
            if fr < fx[0]:
                # Expansion
                gamma = 2.0
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls_made += 1
                if fe < best_fx:
                    best_fx = fe
                    best_x = xe.copy()
                if fe < fr:
                    simplex[-1] = xe
                    fx[-1] = fe
                else:
                    simplex[-1] = xr
                    fx[-1] = fr
            elif fr < fx[-2]:
                simplex[-1] = xr
                fx[-1] = fr
            else:
                # Contraction
                rho = 0.5
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                fc = func(xc)
                calls_made += 1
                if fc < best_fx:
                    best_fx = fc
                    best_x = xc.copy()
                if fc < fx[-1]:
                    simplex[-1] = xc
                    fx[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                        fx[i] = func(simplex[i])
                        calls_made += 1
                        if fx[i] < best_fx:
                            best_fx = fx[i]
                            best_x = simplex[i].copy()
                        if calls_made >= budget:
                            break
            if calls_made >= budget:
                break

        return best_x, best_fx
