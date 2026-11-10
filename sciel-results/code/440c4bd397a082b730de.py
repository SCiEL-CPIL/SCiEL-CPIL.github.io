            if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
                return True
            