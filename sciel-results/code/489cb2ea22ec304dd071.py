                            f_trial, x_trial_valid = evaluate(x_trial)
                            if f_trial is None:
                                break
                            if self._is_nondominated(f_trial, [a[1] for a in archive]):
                                archive = self._add_to_archive(x_trial_valid, f_trial, archive)
                                x_cur = x_trial_valid
                                f_cur = f_trial
    