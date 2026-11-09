<<<<<<< SEARCH
        # Error estimate
        y_err = h * (e[0]*k1 + e[1]*k2 + e[2]*k3 + e[3]*k4 + e[4]*k5 + e[5]*k6)
        err_norm = np.linalg.norm(y_err / (atol + rtol * np.maximum(np.abs(y), np.abs(y + h*b5))))
=======
        # Error estimate
        y_err = h * (e[0]*k1 + e[1]*k2 + e[2]*k3 + e[3]*k4 + e[4]*k5 + e[5]*k6)
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y + h*b5))
        err_norm = np.linalg.norm(y_err / scale)
>>>>>>> REPLACE
